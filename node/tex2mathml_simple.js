#!/data/nsam947/libs/node-v20.13.1-linux-x64/bin/node

/*************************************************************************
 *
 *  Source : https://github.com/Whadup/arxiv_library/blob/master/arxiv_library/compilation/tex2mathml.js
 * 
 *  tex2mml
 *
 *  Uses MathJax to convert a TeX or LaTeX string to a MathML string.
 *
 * ----------------------------------------------------------------------
 *
 *  Copyright (c) 2014 The MathJax Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


const katex = require('katex');
const process = require('process');

var stdin = process.stdin,
    stdout = process.stdout,
    stderr = process.stderr,
    inputChunks = [];

const annotation_regex = /<annotation.*>(.|\s)*<\/annotation>/;

stdin.resume();
stdin.setEncoding('utf8');

stdin.on('data', function (chunk) {
    inputChunks.push(chunk);
});

stdin.on('end', function () {
    var inputJSON = inputChunks.join("");
    var equations = JSON.parse(inputJSON);
    var output = [];

    equations.forEach((latex) => {
        try {
            var mml = katex.renderToString(latex, {
                output: "mathml",
                throwOnError: true,
                strict: "ignore"
            });
            mml = mml.replace(annotation_regex, '');
            output.push(mml);
        } catch (e) {
            if (e instanceof katex.ParseError) {
                var error_message = "Error in LaTeX: " + e.message;
                output.push(error_message);
                process.stderr.write(error_message + "\n");
            } else {
                var error_message = e.message;
                output.push(error_message);
                process.stderr.write(error_message + "\n");
            }
        }
    });

    stdout.write(JSON.stringify(output, null, 4), () => {
        process.exit(0);
    });
});
