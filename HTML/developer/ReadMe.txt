The D3 code used in the production version is a minified version and not human interpretable. In addition to the production version, this folder contains developer version of the code which will help the users in modifying the visuals. This document will help you in accessing and modifying the d3 code.


Requirements:
1.	Node.js (open source and can be downloaded from node website)
2.	Any  browser


Installation:
Copy the developer folder to the local machine and install the modules specified in the package.json file. To install the same open the command prompt from the current folder and run the command 

node package.json


You can also download the node_modules folder available which contains all the libraries required for this. (If you have done this you do not require installing the above libraries)

The code is available in the files ./src/index.js. The sample set of data is available in the file AppData.js. 

The data format should be same as in the file. You can copy the same from the HTML file generated from the python function.
Once the data is placed into the AppData.js file, you need to run the command

npm run-script dev

which will start the development server. It contains hot-line refresh which will get updated on refresh of the data. Open the browser and go to localhost:8080 to access the page.

Once done with development, run 

npm run-script 

make which will create the minified version of the js code. You just need the index.html and the lib folder to access the page once this step is done. 

