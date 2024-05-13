
import argparse
import preprocessing.tex2xml as tex2xml
import preprocessing.xml2graph as xml2graph


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tx", "--tex2xml", metavar='\b', default=False, type=bool, help="Default False. Download Tex equations, convert to XML and save dataset")
    parser.add_argument("-xg", "--xml2graph", metavar='\b', default=False, type=bool, help="Default False. Load XML, convert to graph and save dataset")

    args = parser.parse_args()

    if args.tex2xml:
        tex2xml.main()

    if args.xml2graph:
        xml2graph.main()