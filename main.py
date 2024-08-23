
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
    parser.add_argument("-pl", "--plot", action="store_true", help="Default False. Create plots")
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
        mathml = MathmlDataset.MathmlDataset(xml_name,latex_set=latex_set,debug=debug, force_reload=True)
        vocab = VocabBuilder.VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=False, reload_xml_elements=False)
        dataset = GraphDataset.GraphDataset(mathml.xml_dir,vocab, force_reload=True, debug=debug, max_num_nodes=100)
        print(f"Pre-processed the Dataset '{latex_set}', generated a vocab with the method '{vocab_type}', and is saved into '{mathml.xml_dir}'")
        print(f"The generated dataset contains {len(dataset)} graphs")



        print(dataset[-1].x, dataset[-1].tag)


    if args.plot:
        mathml = MathmlDataset.MathmlDataset(xml_name,latex_set=latex_set,debug=debug, force_reload=False)
        vocab = VocabBuilder.VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=False, reload_xml_elements=False)
        dataset = GraphDataset.GraphDataset(mathml.xml_dir,vocab, force_reload=False, debug=debug, max_num_nodes=100)

        graph = dataset.get_graph_list()[-1]
        plot.plot_graph(graph)

    if args.train:
        dim = 512
        method = {
            "onehot": {},
            "embed": {"tag":256,"mi":dim, "mo":dim, "mtext":dim,"mn":dim},
            "linear": {},
            "loss": "cosine",
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
                "debug": debug,
                "force_reload":False,
                "train_edge_features":True
            }
            train.train_model(config)
        else:
            train.main(model_name, latex_set, vocab_type, xml_name, method, epochs, force_reload)
    
    if args.search:
        search.main(model_name, latex_set, vocab_type, xml_name,epochs)

    if args.test:
        # test.test_all_models(model_name)
        test.main(model_name)

    if args.stats:
        # stats.xml_occurences()
        # stats.count_text_occurences_per_tag()
        # stats.extract_data_from_search()
        # plot.plot_hyperparam_search("trained_models/GAE_search_channel_dims")
        # metrics = {
        #     "loss": [], "train_auc": [], "train_ap": [], "train_acc": [], "train_sim": [], 
        #     "val_loss": [], "val_auc": [], "val_ap": [], "val_acc": [], "val_sim": []
        # }
        # plot.plot_training_history(metrics,"something")

        in_path = "/data/nsam947/Freshwater-Modelling/data/ray_results/concat_search_var"
        out_path = "/data/nsam947/Freshwater-Modelling/trained_models/concat_search_var"
        # stats.extract_data_from_search(path)
        # plot.plot_history_from_search("concat_search_var")
        # plot.plot_study("ablation_study")
        # plot.plot_boxplot_hyperparameters("hyperopt_LAST")

        # stats.create_combined_dataframe(in_path,out_path)

        xml_path = "data/pre_processed/default/xml_elements.json"
        # plot.plot_text_frequency_per_tag(xml_path)
        plot.plot_numbers_distribution(xml_path,"num_val_distrib")
        # stats.test_different_feature_scalings()
