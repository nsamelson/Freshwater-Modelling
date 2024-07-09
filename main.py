
import argparse
import random

import numpy as np
import torch
from preprocessing import MathmlDataset, VocabBuilder, GraphDataset
from models.Graph.GraphAutoEncoder import GraphVAE, GraphEncoder, GraphDecoder
from torch_geometric.nn import GraphSAGE, GCNConv, GraphConv

# from .tests import test_proprocessing as test_prepro
import utils.stats as stats
import utils.plot as plot
from models import train, search, test
from torch.utils.data.dataset import random_split


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Actions
    parser.add_argument("-p", "--preprocess", action="store_true", help="Default False. Download Tex equations, convert to XML and save dataset")
    parser.add_argument("-te", "--test", action="store_true", help="Default False. Test things")
    parser.add_argument("-tr", "--train", action="store_true", help="Default False. Train things")
    parser.add_argument("-se", "--search", action="store_true", help="Default False. Search hyperparams")
    parser.add_argument("-st", "--stats", action="store_true", help="Default False. Create stats")
    # Naming things
    parser.add_argument("-ln", "--latex_name", choices=["OleehyO","sample","Pfahler"], help="Name of the latex Set",default="OleehyO")
    parser.add_argument("-vn", "--vocab_name", choices=["concat","combined","split"], help="Name of the vocab method", default="concat")
    parser.add_argument("-xn", "--xml_name", help="Name of the xml dataset", default="default",)
    parser.add_argument("-mn", "--model_name", help="Name of the model for training", default="default",)
    # Params
    parser.add_argument("-fr", "--force_reload", action="store_true", help="Default False. Force reload the preprocessing")
    parser.add_argument("-d", "--debug", action="store_true", help="Default False. debug")

    args = parser.parse_args()

    # Get the args constants
    xml_name = args.xml_name
    latex_set = args.latex_name
    vocab_type = args.vocab_name
    force_reload = args.force_reload
    debug = args.debug
    model_name = args.model_name


    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.preprocess:      
        print("Starting preprocessing...")
        mathml = MathmlDataset.MathmlDataset(xml_name,latex_set=latex_set,debug=debug, force_reload=force_reload)
        vocab = VocabBuilder.VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=force_reload, reload_xml_elements=force_reload)
        dataset = GraphDataset.GraphDataset(mathml.xml_dir,vocab, force_reload=force_reload, debug=debug)
        print(f"Pre-processed the Dataset '{latex_set}', generated a vocab with the method '{vocab_type}', and is saved into '{mathml.xml_dir}'")
        print(f"The generated dataset contains {len(dataset)} graphs")

        # print(dataset[0].x, dataset[0].tag_index)
        # print(vocab.shape())
        graph = dataset[1]
        print(graph.nums)

        embedding_dim = 256
        method = {
            "onehot": ["concat"],
            "embed": [], # "mi","mo","mtext","mn"
            "linear": [],
            "scale": "log"
        }
        encoder = GraphEncoder(embedding_dim,64,16,2,GCNConv)
        decoder = GraphDecoder(embedding_dim,64,16,2,GCNConv)
        model = GraphVAE(encoder, decoder, vocab.shape(), embedding_dim,method ,True)
        
        x = model.embed_x(graph.x,graph.tag_index, graph.pos, graph.nums)
        print(x.shape, graph.edge_index.shape, graph.edge_attr.shape)
        print(graph.edge_attr)

        z = model.encode(x, graph.edge_index, graph.edge_attr)
        print(z.shape)

        x_p = model.decoder.node_decoder(z,graph.edge_index, graph.edge_attr)
        print(x_p.shape)

        edge_recon = model.decoder.edge_features_decoder(z,graph.edge_index)
        print(edge_recon, edge_recon.shape)

        loss = model.recon_full_loss(z, x, graph.edge_index, None, graph.edge_attr, 1,1, 1)
        loss = loss + (1 / graph.num_nodes) * model.kl_loss()
        print(loss)

        print(model.calculate_accuracy(z,x,graph.edge_index,graph.edge_attr))

        # print(x, x.shape)
        # print(x[9])
        # print(x[17])

        # print(x[-1])
        # a = (x == 1).nonzero(as_tuple=False)
        # print(a, a.shape)


    if args.train:
        train.main(model_name)
    
    if args.search:
        search.main()

    if args.test:
        test.main()

    if args.stats:
        # stats.xml_occurences()
        # stats.count_text_occurences_per_tag()
        # stats.extract_data_from_search()
        plot.plot_hyperparam_search("trained_models/GAE_search_channel_dims")
