# import the libraries
import csv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Upload the model and the embeding model
Settings.llm = Ollama(
    model='gemma4:e2b',
    request_timeout=300.0,
    timeout=300.0,
    keep_alive='60m',
)

print('Warming up model...')
Settings.llm.complete('hi')
print('Model warm')

Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text')

print('Models configured')
print('   LLM      : gemma4:e2b')
print('   Embedder : nomic-embed-text')

# load the data and metadata managment 
def build_nodes_from_csv(filepath: str) -> list:
    nodes = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:

            # TEXT: what the EMBEDDER sees
            text = (
                f"{row['name']} is a {row['category']} product "
                f"from {row['supplier']}, priced at ${row['price']}. "
                f"It has a customer rating of {row['rating']} out of 5."
            )

            stock   = int(row['stock_quantity'])
            reorder = int(row['reorder_point'])
            sales   = int(row['last_month_sales'])
            rating  = float(row['rating'])
            lead    = int(row['lead_time_days'])

            # METADATA: what the LLM sees
            metadata = {
                'product_id'       : row['product_id'],
                'name'             : row['name'],
                'category'         : row['category'],
                'supplier'         : row['supplier'],
                'price'            : float(row['price']),
                'stock_quantity'   : stock,
                'rating'           : rating,
                'lead_time_days'   : lead,
                'reorder_point'    : reorder,
                'last_month_sales' : sales,
                'is_out_of_stock'  : stock == 0,
                'needs_reorder'    : stock < reorder,
                'stock_gap'        : reorder - stock,
            }

            node = TextNode(
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=[
                    'product_id', 'stock_quantity', 'reorder_point',
                    'last_month_sales', 'lead_time_days',
                    'stock_gap', 'needs_reorder', 'is_out_of_stock',
                ],
                excluded_llm_metadata_keys=['product_id', 'stock_gap'],
            )
            nodes.append(node)
    return nodes

nodes = build_nodes_from_csv('products.csv')
print(f'Built {len(nodes)} nodes')
print('   Products:', [n.metadata['name'] for n in nodes])

# inspect one node in details
INSPECT_INDEX = 0

node = nodes[INSPECT_INDEX]
sep = '-' * 55

print(sep)
print(f"  PRODUCT : {node.metadata['name']}")
print(sep)

print("\nTEXT (what gets embedded):")
print('   ', node.get_content(metadata_mode=MetadataMode.NONE))

print("\nEMBED view (text + embed-allowed metadata):")
print(node.get_content(metadata_mode=MetadataMode.EMBED))

print("\nLLM view (what appears inside the prompt context):")
print(node.get_content(metadata_mode=MetadataMode.LLM))

print("\nFULL RAW METADATA:")
for k, v in node.metadata.items():
    note = ''
    if k == 'is_out_of_stock' and v:  note = '  <- OUT OF STOCK'
    if k == 'needs_reorder'   and v:  note = '  <- NEEDS REORDER'
    print(f'   {k:<22} = {v}{note}')

# Create the Agent role and the prompt template
PRODUCT_TRACKER_PROMPT = PromptTemplate(
    """
You are a PRODUCT TRACKER AGENT — a specialist in inventory health and product performance.

YOUR RESPONSIBILITIES:
- Identify products that are out of stock or need reordering
- Highlight top-performing products based on sales volume and customer ratings
- Flag supply risks: high lead_time_days combined with low stock_quantity
- Deliver clear, structured, actionable reports — no guessing or inventing numbers

LABELING RULES (apply these consistently):
  OUT OF STOCK   -> stock_quantity = 0
  NEEDS REORDER  -> stock_quantity < reorder_point
  TOP PERFORMER  -> rating >= 4.6 AND last_month_sales > 200
  SUPPLY RISK    -> lead_time_days >= 7 AND stock_quantity < reorder_point

OUTPUT FORMAT:
  Always end your response with a short "PRIORITY ACTIONS" section
  listing the top 3 things that need immediate attention.

STRICT RULES:
  1. Only use numbers from the context below — never invent or estimate
  2. If a product is not in the context, do not mention it
  3. Be concise — use bullet points where possible

PRODUCT DATA:
{context_str}

QUESTION: {query_str}

YOUR REPORT:
"""
)

print('Product Tracker prompt defined')
print('\n--- PROMPT PREVIEW ---')
print(PRODUCT_TRACKER_PROMPT.template)

# build the vector index and the query engine
print('Building vector index...')
index = VectorStoreIndex(nodes, show_progress=True)
print('\nIndex ready — all products embedded')

tracker = index.as_query_engine(
    text_qa_template=PRODUCT_TRACKER_PROMPT,
    similarity_top_k=15,
)

print('Product Tracker Agent is live')
print('   similarity_top_k = 15 (all products in context)')
print('   prompt           = PRODUCT_TRACKER_PROMPT')
print('   model            = gemma4:e2b')

# test the agent with a question
print('Which products are performing best?\n')

response = tracker.query(
    'Which products are performing the best? '
    'Consider both rating and last_month sales. '
    'Explain why each one stands out.'
)
print(response)