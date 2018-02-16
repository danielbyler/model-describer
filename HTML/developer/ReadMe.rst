.. -*- mode: rst -*-

UI Developer Toolkit:
=====================

WhiteBox consists of two discrete parts- calculations done in Python to create the `underlying data <https://github.com/Data4Gov/WhiteBox_Production/tree/master/whitebox>`_ and the D3 code that forms the charts and text from that data. The production D3 code WhiteBox is minified which makes it difficult for potential contributors to read and edit. To make it easier for others to contribute to this project, this folder contains a developer friendly version of the code which will help developers modify the visuals. 

The following steps and requirements show how to create the necessary minified code for both of WhiteBox’s main chart types (error and impact). 

Requirements:
-------------

1. `Node.js <https://nodejs.org/en/>`_  
2. Any browser with javascript enabled
3. Packages listed in packages.json in each of the parent folders  

Developer Friendly Code:
------------------------
To start with the developement, few javascript libraries are required which were added to the package.json file. Open the command line and switch to the current directory. Run the following command

```
npm install
```

which will create a folder named **node_modules** and install all the necessary libraries in it.



The developer-friendly D3 code is available at ./src/index.js inside of each folder. A sample dataset is available in the file ./src/AppData.js. 

For consistency and the ability to integrate with the backend, it is crucial that your D3 code take this data in its existing format (unless you are also modifying the backend).

To test your code, first run this line of code in the command line:

```
npm run-script dev
```

This will start the development server. It contains hot-line refresh which will get updated on refresh of the code. Open the browser and go to localhost:8080 to access the page.

Once done with development, run 
```
npm run-script make 
```
This will create the minified version of the js code and write it to App.min.js file in the ./lib folder. Open the index.html to see the results.

Happy contributing!  
