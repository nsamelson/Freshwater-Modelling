
import argparse
from preprocessing import MathmlDataset, VocabBuilder, GraphDataset
# from .tests import test_proprocessing as test_prepro
import utils.stats as stats
import utils.plot as plot
from models import train, search, test
from torch.utils.data.dataset import random_split


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pr", "--preprocess", action="store_true", help="Default False. Download Tex equations, convert to XML and save dataset")
    parser.add_argument("-xg", "--xml2graph", action="store_true", help="Default False. Load XML, convert to graph and save dataset")
    parser.add_argument("-st", "--stats", action="store_true", help="Default False. Create stats")
    parser.add_argument("-te", "--test", action="store_true", help="Default False. Test things")
    parser.add_argument("-tr", "--train", action="store_true", help="Default False. Train things")
    parser.add_argument("-se", "--search", action="store_true", help="Default False. Search hyperparams")



    args = parser.parse_args()

    if args.preprocess:
        
        xml_name = "debug"
        mathml = MathmlDataset.MathmlDataset(xml_name,latex_set="OleehyO",debug=True, force_reload=False)
        vocab = VocabBuilder.VocabBuilder(xml_name,vocab_type="split", debug=True,reload_vocab=False,reload_xml_elements=False)

        dataset = GraphDataset.GraphDataset(mathml.xml_dir,vocab, force_reload=False, debug=True)

        train,val, test = dataset.split()
        print(train[0].x)
        print(train.get_graph_list()[0].nodes(data=True))
        # print(dataset.get_graph_list()[0])



    # if args.xml2graph:
    #     xml2graph.main()
    
    if args.stats:
        # stats.xml_occurences()
        # stats.count_text_occurences_per_tag()
        # stats.extract_data_from_search()
        plot.plot_hyperparam_search("trained_models/GAE_search_channel_dims")

    if args.train:
        train.main()
    
    if args.search:
        search.main()

    if args.test:
        test.main()