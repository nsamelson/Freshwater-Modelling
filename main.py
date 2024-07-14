
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
from torch_geometric.utils import negative_sampling

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

    parser.add_argument("-e", "--epochs", help="Number of epochs for training", default=200, type=int)
    # parser.add_argument("-oh", "--one_hot", help="OneHot method(s) as list", nargs='*', default=[], choices=["concat","tag","pos"])
    # parser.add_argument("-em", "--embed", help="Embed method(s) as list", nargs='*', default=[], choices=["tag","concat","combined","mi","mo","mtext","mn"])
    # parser.add_argument("-li", "--linear", help="Linear method(s) as list", nargs='*', default=[], choices=["mn"])
    # parser.add_argument("-lo", "--loss", help="Loss function", default="cross_entropy", choices=["cross_entropy","mse","cosine"])

    args = parser.parse_args()

    # Get the args constants
    latex_set = args.latex_name
    vocab_type = args.vocab_name
    xml_name = args.xml_name
    model_name = args.model_name
    force_reload = args.force_reload
    debug = args.debug
    epochs = args.epochs 
    # one_hot = args.one_hot
    # embed = args.embed
    # linear = args.linear
    # loss = args.loss



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
        # graph = dataset[1]
        # print(graph.nums)

        # embedding_dim = 0
        # method = {
        #     "onehot": {},
        #     "embed": {"concat":256,}, # "mi","mo","mtext","mn"
        #     "linear": {},
        #     "scale": "log",
        #     "loss": "mse"
        # }
        # embedding_dim = sum(method["onehot"].values()) + sum(method["embed"].values()) + sum(method["linear"].values())

        # encoder = GraphEncoder(embedding_dim,64,16,2,GCNConv)
        # decoder = GraphDecoder(embedding_dim,64,16,2,GCNConv,edge_dim=1 )
        # model = GraphVAE(encoder, decoder, vocab.shape(), embedding_dim,method ,True)
        
        # x = model.embed_x(graph.x,graph.tag_index, graph.pos, graph.nums)
        # print(x.shape, graph.edge_index.shape, graph.edge_attr.shape)
        # print(graph.edge_attr)

        # z = model.encode(x, graph.edge_index, graph.edge_attr)
        # print(z.shape)

        # x_p = model.decoder.node_decoder(z,graph.edge_index, graph.edge_attr)
        # print(x_p.shape)

        # edge_recon = model.decoder.edge_decoder(z,graph.edge_index)
        # print(edge_recon, edge_recon.shape)

        # loss = model.recon_full_loss(z, x, graph.edge_index, None, graph.edge_attr, 1,1, 1)
        # loss = loss + (1 / graph.num_nodes) * model.kl_loss()
        # print(loss)

        # neg_edge_index = negative_sampling(graph.edge_index, graph.size(0))
        # accuracy = model.calculate_accuracy(z,graph.edge_index,neg_edge_index, graph.x, graph.edge_attr,)
        # print(accuracy)
        # # stuff = model.test(z,x,graph.edge_index,neg_edge_index, graph.x, graph.edge_attr)
        # # # auc, ap = model.test(z, graph.edge_index, neg_edge_index)
        # # print("NODE, ADJ, EDGES: ",stuff)

        # node_sim, adj_sim, edge_sim = model.calculate_similarity(z,x, graph.edge_index, graph.edge_attr)
        # print("SIM: ", node_sim,adj_sim, edge_sim)

        # node, adjency, edges = model.decode_all(z,graph.edge_index)
        # print(graph.x)
        # print(node)


    if args.train:

        method = {
            "onehot": {"concat":256},
            "embed": {},
            "linear": {},
            "loss": "cross_entropy",
            "scale": "log",
        }
        if debug:
            config = {
                "model_name": "debug",
                "xml_name": "debug",
                "num_epochs": 1,
                "latex_set":"OleehyO",
                "vocab_type":"concat",
                "method": method,
                "debug": debug
            }
            train.train_model(config)
        else:
            train.main(model_name, latex_set, vocab_type, xml_name, method, epochs, force_reload)
    
    if args.search:
        search.main()

    if args.test:
        test.main()

    if args.stats:
        # stats.xml_occurences()
        # stats.count_text_occurences_per_tag()
        # stats.extract_data_from_search()
        # plot.plot_hyperparam_search("trained_models/GAE_search_channel_dims")
        metrics = {
            "loss": [], "train_auc": [], "train_ap": [], "train_acc": [], "train_sim": [], 
            "val_loss": [], "val_auc": [], "val_ap": [], "val_acc": [], "val_sim": []
        }
        plot.plot_training_history(metrics,"something")
