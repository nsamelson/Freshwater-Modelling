import os
import json
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG)

def call_js(latex_equations, paper_id=""):
    try:
        p, _ = os.path.split(__file__)
        script_path = os.path.join(p, "tex2mathml_simple.js")

        # Add the directory where node is installed to PATH
        env = os.environ.copy()
        node_bin_dir = "/data/nsam947/libs/node-v20.13.1-linux-x64/bin"
        env["PATH"] = node_bin_dir + os.pathsep + env["PATH"]

        # logging.debug("Running tex2mathml.js with environment PATH: {}".format(env["PATH"]))

        result = subprocess.run(
            [script_path],
            input=json.dumps(latex_equations),
            cwd=os.path.join(p),
            env=env,
            universal_newlines=True,
            text=True,
            capture_output=True,
            timeout=120
        )

        # logging.debug("stderr output: {}".format(result.stderr))
        # logging.debug("stdout output: {}".format(result.stdout))

        raw_output = result.stdout.strip()
        # logging.debug("Raw output before JSON parsing: {}".format(raw_output))

        if result.stderr:
            logging.warning("Unexpected error in tex2mathml.js (Arxiv ID: {}):".format(paper_id) + result.stderr)

        if raw_output:
            try:
                result_data = json.loads(raw_output)
                return result_data
            except json.JSONDecodeError as e:
                logging.error("JSON decoding failed: {}".format(e))
                return None

    except subprocess.TimeoutExpired:
        logging.warning("Timeout for paper {}: \n".format(paper_id) + "\n")
        return False
    except Exception as e:
        logging.error("Error calling tex2mathml.js: {}".format(e))
        return False


if __name__ == "__main__":
    # Example usage of call_js
    latex_equations = ["f(x) = x^2", "a^2 = 3"]
    result = call_js(latex_equations)
    print(result)
