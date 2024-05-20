
import argparse
import preprocessing.tex2xml as tex2xml
import preprocessing.xml2graph as xml2graph
import utils.stats as stats


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tx", "--tex2xml", metavar='\b', default=False, type=bool, help="Default False. Download Tex equations, convert to XML and save dataset")
    parser.add_argument("-xg", "--xml2graph", metavar='\b', default=False, type=bool, help="Default False. Load XML, convert to graph and save dataset")
    parser.add_argument("-s", "--stats", metavar='\b', default=False, type=bool, help="Default False. Create stats")

    args = parser.parse_args()

    if args.tex2xml:
        tex2xml.main()

    if args.xml2graph:
        xml2graph.main()
    
    if args.stats:
        # stats.xml_occurences()
        stats.count_text_occurences_per_tag()
