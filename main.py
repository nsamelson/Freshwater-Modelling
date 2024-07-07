
import argparse
from preprocessing import MathmlDataset, VocabBuilder, GraphDataset
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


    if args.preprocess:      
        mathml = MathmlDataset.MathmlDataset(xml_name,latex_set=latex_set,debug=debug, force_reload=force_reload)
        vocab = VocabBuilder.VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=force_reload, reload_xml_elements=force_reload)
        dataset = GraphDataset.GraphDataset(mathml.xml_dir,vocab, force_reload=force_reload, debug=debug)

    if args.train:
        train.main()
    
    if args.search:
        search.main()

    if args.test:
        test.main()

    if args.stats:
        # stats.xml_occurences()
        # stats.count_text_occurences_per_tag()
        # stats.extract_data_from_search()
        plot.plot_hyperparam_search("trained_models/GAE_search_channel_dims")
