//App helper library
//REQUIRES D3 to be loaded on page to function
export default class AppHelper {

    //setting up the title
    setTitle(el, text) {
        el.selectAll('*').remove();
        var title = el.append('span').html(text);
    }
    
    setLabel(el, text) {
        el.selectAll('*').remove();
        var title = el.append('span').html(text);
    }
    
    setSource(el, text) {
        el.selectAll('*').remove();
        el.html(text);
    }

    //function to format the labels
    formatLabel(label) {
        label = d3.format("." + 3 + "f")(label)
        label = label.replace(/\.0$/, '')
        return label
    }
    function formatLabelNew(label) {
        var format = d3.format(".2s")
        console.log(format(label))
        labelStr = label+""
        labelDec = "."+labelStr.split(".")
        labelDec = (labelDec).toPrecision(3)
        console.log(labelDec)
        return Math.floor(label)+labelDec
    }
    //function to create a svg element
    createChart(parent, width) {
        var chart = parent.append('svg');
        chart.attr('width', width);
        return chart;
    }


    //Generic
    getWidth() {
        return window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    }
    //function to sort the dictionary
    sortDictionary(a, b) {
        var nameA = a.name.toLowerCase(),
            nameB = b.name.toLowerCase()
        if (nameA < nameB) //sort string ascending
            return -1
        if (nameA > nameB)
            return 1
        return 0 //default return value (no sorting)
    }
    //function to sort a list based on alpha numeric characters
    sortAlphaNum(a, b) {
        var reA = /[^a-zA-Z]/g;
        var reN = /[^0-9]/g;
        var aA = a.replace(reA, "");
        var bA = b.replace(reA, "");
        if (aA === bA) {
            var aN = parseInt(a.replace(reN, ""), 10);
            var bN = parseInt(b.replace(reN, ""), 10);
            return aN === bN ? 0 : aN > bN ? 1 : -1;
        } else {
            return aA > bA ? 1 : -1;
        }
    }
}
