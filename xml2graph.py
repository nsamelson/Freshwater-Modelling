import math
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

# Parse the MathML equation and account for namespaces
ns = {'mathml': "http://www.w3.org/1998/Math/MathML"}
tree = ET.parse('out/test.xml')
root = tree.getroot()




# Create a directed graph
G = nx.DiGraph()

def create_node(element, parent=None):
    # remove the namespace
    if element.tag.startswith('{'):
        element.tag = element.tag.split('}', 1)[-1]


    node_id = element.tag #+ "_" + element.text if element.text else element.tag
    
    # if "math" not in node_id:        
    G.add_node(node_id, label=node_id)

    if parent and node_id != 'math':
        G.add_edge(parent, node_id)


    for child in element:
        create_node(child, node_id)

    if element.text:
        G.add_node(element.text,label=element.text)
        G.add_edge(node_id,element.text)

# Create graph nodes and edges
create_node(root, 'math')

# Print graph nodes and edges
print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges())

# Draw the graph using NetworkX's built-in drawing functions
pos = nx.spring_layout(G,k = 1/math.sqrt(len(G.nodes.values())))  # Compute graph layout

nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos=pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos=pos, font_size=10, font_family='sans-serif')

plt.axis('off')

# Show the graph
plt.title('MathML Structure Graph')
plt.savefig('out/graph.jpg', format='jpeg', dpi=300) 