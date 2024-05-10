import math
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

def main():
    """
    Generate a graph from mathml XML equations. 
    Source : https://github.com/Whadup/arxiv_learning/blob/ecml/arxiv_learning/data/load_mathml.py
    """
    # Parse the MathML equation and account for namespaces
    tree = ET.parse('out/test.xml')
    root = tree.getroot()

    
    base =  {'type': 'math', 'children': [
        {'type': 'mrow', 'children': [
            {'type': 'msup', 'children': [
                {'type': 'mi', 'content': 'A', 'attributes': []}, 
                {'type': 'mn', 'content': '2', 'attributes': []}
            ], 'content': ''}, 
            {'type': 'mo', 'content': '+', 'attributes': []}
        ], 'content': ''}, 
        {'type': 'mrow', 'children': [
            {'type': 'msup', 'children': [
                {'type': 'mi', 'content': 'B', 'attributes': []}, 
                {'type': 'mn', 'content': '2', 'attributes': []}
            ], 'content': ''}, 
            {'type': 'mo', 'content': '=', 'attributes': []}
        ], 'content': ''}, 
        {'type': 'mrow', 'children': [
            {'type': 'msup', 'children': [
                {'type': 'mi', 'content': 'C', 'attributes': []}, 
                {'type': 'mn', 'content': '2', 'attributes': []}
            ], 'content': ''}
        ], 'content': ''}
    ], 'content': ''}

    
    # def process(d):
    #     """Convert XML-Structure to Python dict-based representation"""
    #     children = [x for x in d]
    #     if children:
    #         children = []
    #         for x in d:
    #             tag = rn(x.tag)
    #             if tag == "annotation":
    #                 #skip the latex annotation
    #                 continue
    #             children.append(process(x))
    #         return dict(type=rn(d.tag),
    #                     children=children,
    #                     content="" if d.text is None else d.text)
    #     return dict(type=rn(d.tag),
    #                 content="" if d.text is None else d.text)
    
    # graph = process(root)
    # print(graph)
    # Initialize a directed graph and a global variable 'uid'
    G = nx.MultiDiGraph()
    global uid
    uid = 0
    

    def create_node(element,parent_uid=0):
        global uid

        # Adding parent node "Math"
        if len(G.nodes) == 0:
            G.add_node(uid,tag=rn(element.tag),label=element.text)
            uid += 1

        for child in element:
            tag = rn(child.tag)

            G.add_node(uid, tag=tag, label=child.text)
            G.add_edge(parent_uid, uid)
            uid += 1

            children = [x for x in child]
            if children:
                create_node(child,uid-1)



    # Create graph nodes and edges
    create_node(root)

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


    # pyg_graph = from_networkx(G)



def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")