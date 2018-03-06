//Importing helper functions
import AppHelper from './AppHelper.js';

//Importing Data
import {
    AppData
} from './AppData.js';
console.log(AppData)

//SASS Import to DOM at runtime
require('./styles/index.scss');

//Creatign helper functions object
var appHelper = new AppHelper();


//Creating the layout for all the elements
var main = d3.select('#App');
//Main Heading
var heading = main.append("div").attr('class', "heading").append("a").attr("href", "https://github.com/DataScienceSquad/WhiteBox_Production").html("White Box - Prediction Error By Variable")

var heading2 = main.append("div").attr('class', "subheading").append("a").attr("href", "https://github.com/DataScienceSquad/WhiteBox_Production/wiki/How-to-interpret-WhiteBox-charts").html("How to interpret WhiteBox Charts")
//Heat map and the type dropdown
var summary = main.append("div").attr("class", "summary")
var heatMapContainer = summary.append('div').attr('class', 'heatMapContainer')
var filterContainer = summary.append('div').attr('class', 'filt-cont')


//Main chart container
var mainApp = main.append('div').attr('class', 'Chartcontainer')
var colorList = ["#86BC25", "#00A3E0", "#00ABAB", "#C4D600", "#2C5234", "#9DD4CF", "#004F59", "#62B5E5", "#43B02A", "#0076A8", "#9DD4CF", "#012169"]
var margin = {
    top: 20,
    right: 15,
    bottom: 60,
    left: 60
}
var showingTooltip = false;


//Adding tooltip and setting up default properties
var tooltip = d3.select("body").append("div")
    .attr("id", "tooltip")
    .style("opacity", 0)
d3.select('body').on('click', function () {
    if (showingTooltip) {
        tooltip.style("opacity", 0);
        showingTooltip = false;
    }
});

//Intializing global variables
var permKeys = ['predictedYSmooth', 'groupByVarName', 'groupByValue', 'highlight', 'errNeg', 'errPos']
var percFlag = true;
//Application
var heatMapData = {}
var colorDict = {}
var width, svg, svgHeat, globalHeightOffset, percentScale, responses;
var groupBy = ''
var heatMapFilters = []
var metaData = {}


//Creating svg container for Heat map


svgHeat = appHelper.createChart(heatMapContainer, width)


var yVar = "Quality"
var percList = [10, 50, 90]



// Heat map onclick filter funtion
function heatMapClick(d, i) {
    var filt = d.data.name

    if (heatMapFilters.indexOf(filt) >= 0) {
        heatMapFilters.splice(heatMapFilters.indexOf(filt), 1)
        d3.select(this).attr("fill", colorDict['heat'][filt])
    } else {
        heatMapFilters.push(filt)
        d3.select(this).attr("fill", "#A7A8AA")
    }
    mainApp.selectAll("*").remove()
    for (var ind = 0; ind < AppData.length; ind++) {
        loopVariables(ind)
    }

}

//Toggle percentile lines function
function togglePercentile() {
    percFlag = document.getElementById("percentileCheck").checked
    d3.select(".Chartcontainer").selectAll("*").remove()
    width = appHelper.getWidth();

    intializeTreeMap(AppData[0]['Data'][0]['groupByVarName'])
}


window.togglePercentile = togglePercentile


//Intializing the filter dropdown for group by varaible and auto populating the list
function intializeFilterDropdown() {
    var filterList = prepareFilterData()
    var typeEl = filterContainer.append("div")
        .attr("class", "dropdown")
    typeEl.append("button")
        .attr("class", "dropbtn")
        .html("<b>Group By:</b> " + groupBy)
        .append("span")
        .style("float", "right")
        .html(" ▼")

    var optionList = typeEl.append("div")
        .attr("class", "dropdown-content")

    optionList.selectAll("a")
        .data(filterList)
        .enter().append("a")
        .html(function (d) {
            return d
        })
        .attr("onclick", function (d) {
            return "intializeTreeMap('" + d + "')"
        })
    var percCheck = filterContainer.append("div")
        .attr("class", "perc-check")
        .html("<input id='percentileCheck' class='perc-checkinput' checked='true' onchange='togglePercentile()' type='checkbox'><label for='percentileCheck'>Percentiles</label>")

}

//Function to show the tooltip on hover of treemap
function showToolTipTreeMap(d, i) {
    setTimeout(function () {
        showingTooltip = true;
    }, 300)
    var statEach = metaData['statData'].filter(function (s) {
        return s['groupByVarName'] == groupBy && s['groupByValue'] == d.data.name
    })[0]
    var percEach = metaData['proportion'].filter(function (s) {
        return s['name'] == d.data.name
    })[0]
    tooltip.style("opacity", 1)
        .style("left", d3.event.pageX + "px")
        .style("top", d3.event.pageY + "px")
        .style("color", colorDict['heat'][d])
        .html("Group <b>" + statEach['groupByValue'] + " </b> is " + percEach['perc'] + "% of the data and has an average error of <b>" + appHelper.formatLabel(statEach[metaData['ErrType']]) + "</b>")
}


//Function to show the tooltip on hover of treemap
function moveToolTipTreeMap(d, i) {
    setTimeout(function () {
        showingTooltip = true;
    }, 300)
    var statEach = metaData['statData'].filter(function (s) {
        return s['groupByVarName'] == groupBy && s['groupByValue'] == d.data.name
    })[0]
    var percEach = metaData['proportion'].filter(function (s) {
        return s['name'] == d.data.name
    })[0]
    tooltip.style("opacity", 1)
        .style("left", d3.event.pageX + "px")
        .style("top", d3.event.pageY + "px")
        .style("color", colorDict['heat'][d.data.name])
        .html("Group <b>" + statEach['groupByValue'] + " </b> is " + appHelper.formatLabel(percEach['perc']) + "% of the data and has an average error of <b>" + appHelper.formatLabel(statEach[metaData['ErrType']]) + "</b>")
}

//Function to hide the tooltip on mouse out of treemap
function hideToolTipTreeMap(d, i) {
    tooltip.style("opacity", 0)
}


//Function to create treemap
function intializeTreeMap(type) {

    colorDict['heat'] = {}
    groupBy = type
    createColorDict()
    d3.select(".dropbtn").html("<b>Group By:</b> " + groupBy).append("span")
        .style("float", "right")
        .html(" ▼")
    var data = getTreeMapData(groupBy)
    var margin = {
        top: 50,
        right: 15,
        bottom: 30,
        left: 60
    }
    svgHeat.selectAll('*').remove();
    var width = appHelper.getWidth() * 0.8 - margin.left * 0.1 - margin.right - 0.1 * appHelper.getWidth();
    var height = 200 - margin.top - margin.bottom;
    svgHeat.attr("width", width)
    svgHeat.attr("height", height)
    var fader = function (color) {
            return d3.interpolateRgb(color, "#fff")(0.2);
        },
        color = d3.scaleOrdinal(d3.schemeCategory20.map(fader)),
        format = d3.format(",d")

    var treemap = d3.treemap()
        .tile(d3.treemapResquarify)
        .size([width, height - 20])
        .round(true)
        .paddingInner(2);

    function sumByCount(d) {
        return d.children ? 0 : 1;
    }

    function sumBySize(d) {
        return d.size;
    }
    var root = d3.hierarchy(data)
        .eachBefore(function (d) {
            d.data.id = (d.parent ? d.parent.data.id + "." : "") + d.data.name;
        })
        .sum(sumBySize)
        .sort(function (a, b) {
            return b.data.name - a.data.name;
        });
    treemap(root);
    metaData['proportion'] = root.data.children
    var sumProportion = 0
    for (var p in metaData['proportion']) {
        sumProportion += metaData['proportion'][p]['size']
    }
    for (var p in metaData['proportion']) {
        metaData['proportion'][p]['perc'] = metaData['proportion'][p]['size'] * 100 / sumProportion
    }
    /*var title = svgHeat.selectAll(".title")
        .data(["Proportion by each variable"])
        .enter().append("text")
        .attr("y", "5px")
        .attr("class", "title")
        .text(function (d) {
            return d
        })
        .style("font-size", "16px")*/
    
    var cellGroup = svgHeat.append("g").attr("class", "cell-group").attr("transform", "translate(0,0)")
    var cell = cellGroup.selectAll("g")
        .data(root.leaves())
        .enter().append("g")
        .attr("transform", function (d) {
            return "translate(" + d.x0 + "," + d.y0 + ")";
        });
    cell.append("rect")
        .attr("id", function (d) {
            return d.data.id + " brdr";
        })
        .attr("x", -2)
        .attr("y", -2)
        .attr("class", "borderpath")
        .attr("width", function (d) {
            return d.x1 - d.x0 + 4;
        })
        .attr("height", function (d) {
            return d.y1 - d.y0 + 4;
        })
        .attr("fill", function (d) {
            return "#53565A";
        })
        .style("cursor", "pointer");

    cell.append("rect")
        .attr("id", function (d) {
            return d.data.id;
        })
        .attr("class", "rect")
        .attr("width", function (d) {
            return d.x1 - d.x0;
        })
        .attr("height", function (d) {
            return d.y1 - d.y0;
        })
        .attr("fill", function (d) {
            return colorDict['heat'][d.data.name];
        })
        .on("click", heatMapClick)
        .on("mouseenter", showToolTipTreeMap)
        .on("mouseout", hideToolTipTreeMap)
        .on("mousemove", moveToolTipTreeMap)
        .style("cursor", "pointer");


    cell.append("text")
        .attr("clip-path", function (d) {
            return "url(#clip-" + d.data.id + ")";
        })
        .attr("dx", 10)
        .attr("dy", 20)
        .text(function (d) {
            if (d.x1 - d.x0 > 100) {
                var statEach = metaData['statData'].filter(function (s) {
                    return s['groupByVarName'] == groupBy && s['groupByValue'] == d.data.name
                })[0]
                
                var percEach = metaData['proportion'].filter(function (s) {
                    return s['name'] == d.data.name
                })[0]
                return "Category: " + statEach['groupByValue'] + " (" + appHelper.formatLabel(percEach['perc']) + "%)";
            } else {
                return ""
            }
        })
        .style("fill", "#fff")
    cell.append("text")
        .attr("clip-path", function (d) {
            return "url(#clip-" + d.data.id + ")";
        })
        .attr("dx", 10)
        .attr("dy", 40)
        .text(function (d) {
            if (d.x1 - d.x0 > 100) {
                var statEach = metaData['statData'].filter(function (s) {
                    return s['groupByVarName'] == groupBy && s['groupByValue'] == d.data.name
                })[0]
                return "Average Error: " + appHelper.formatLabel(statEach[metaData['ErrType']]);
            } else {
                return ""
            }
        })
        .style("fill", "#fff");





    d3.selectAll("input")
        .data([sumBySize, sumByCount], function (d) {
            return d ? d.name : this.value;
        })
    //.on("change", changed);

    mainApp.selectAll("*").remove()
    for (var ind = 0; ind < AppData.length; ind++) {
        loopVariables(ind)
    }

}

//Function to extract the data for Treemap and format it to form required
function getTreeMapData(type) {
    var heatMapData = {
        name: 'heatMap',
        children: []
    }
    var sample = AppData[0]['Data']
    //console.log(AppData,metaData)
    var filteredSample = sample.filter(function (d) {
        return d['groupByVarName'] == type
    })
    var statDataF = metaData['statData'].filter(function (d) {
        return d['groupByVarName'] == type 
    })
    var nestedSample = d3.nest()
        .key(function (d) {
            return d['groupByValue']
        })
        .rollup(function (ids) {
            return ids.length
        })
        .entries(statDataF)
    nestedSample.forEach(function (d) {
        var temp = {}
        temp['name'] = d.key
        temp['size'] = statDataF.filter(function (k) {
            return k['groupByValue'] == d.key
        })[0]['Total']
        heatMapData['children'].push(temp)
    })

    heatMapData['children'] = heatMapData['children'].sort(appHelper.sortDictionary)
    return heatMapData
}
window.intializeTreeMap = intializeTreeMap
//preparing data for filters
function prepareFilterData() {
    var filterList = []
    AppData.forEach(function (group) {
        group['Data'].forEach(function (record) {
            if (filterList.indexOf(record['groupByVarName']) < 0) {
                filterList.push(record['groupByVarName'])
            }
        })
    })
    return filterList
}



//Function to get the xth percentile of the data given
function getXthPercentaile(dataList, x, cat, varX) {

    var varDict = metaData['percData'].filter(function (d) {
        return d.variable == varX
    })[0]

    var typeDict = varDict['percentileList'].filter(function (d) {
        return d.groupByVar == cat
    })[0]
    var percDict = typeDict['percentileValues'].filter(function (d) {
        return d.percentiles == x + "%"
    })[0]
    return percDict.value
    /*var filteredData = dataList.filter(function (d) {
        return d['groupByValue'] == cat
    })
    if (x == 100) {
        var percentileValue = filteredData[filteredData.length - 1][varX]
        return appHelper.formatLabel(percentileValue)
    } else {
        var percRank = (x / 100) * (filteredData.length + 1)
        var percRankInt = parseInt(percRank)
        if(percRankInt){
            var percentileValue = filteredData[percRankInt-1][varX]
        } 
        else{
        var percentileValue = filteredData[percRankInt][varX]
        }
        return appHelper.formatLabel(percentileValue)
    }*/

}
function getXthPercentaileGlobal(dataList, x, cat, varX) {
    var varDict = metaData['percGData'].filter(function (d) {
        return d.variable == varX & d.percentile == x+"%"
    })[0]

    return varDict.value
    /*var filteredData = dataList.filter(function (d) {
        return d['groupByValue'] == cat
    })
    if (x == 100) {
        var percentileValue = filteredData[filteredData.length - 1][varX]
        return appHelper.formatLabel(percentileValue)
    } else {
        var percRank = (x / 100) * (filteredData.length + 1)
        var percRankInt = parseInt(percRank)
        if(percRankInt){
            var percentileValue = filteredData[percRankInt-1][varX]
        } 
        else{
        var percentileValue = filteredData[percRankInt][varX]
        }
        return appHelper.formatLabel(percentileValue)
    }*/

}
//function to return the quartile data from the data provided for continous variables
function createQuartileData(dataList, varX, cats) {
    
    if (cats.length==0){
        return 0
    }
    var varRange = d3.extent(dataList.map(function (d) {
        return d[varX]
    }))

    var statData = {}
    var returnDict = {}
    returnDict['overEst'] = {}
    returnDict['underEst'] = {}
    statData['quartiles'] = []
    var diff = d3.quantile(varRange, 0.25) - d3.min(varRange)
    //console.log(cats)

    for (var c in cats) {
        var cat = cats[c]
        returnDict['underEst'][cat] = ''
        returnDict['overEst'][cat] = ''
        for (var i = 1; i <= 4; i++) {
            
            var quartile = {}
            quartile['quarter'] = i
            quartile['group'] = cat
            //console.log(varX,AppData)
            var low = getXthPercentaile(dataList, (i - 1) * 25, cat, varX)
            var high = getXthPercentaile(dataList, (i) * 25, cat, varX)
            quartile['range'] = [low, high]
            var filtData = dataList.filter(function (d) {
                return d[varX] >= low && d[varX] <= high && d['groupByValue'] == cat
            })

            quartile['averageImpact'] = d3.mean(filtData.map(function (d) {
                return d['predictedYSmooth']
            }))
            quartile['medianImpact'] = d3.median(filtData.map(function (d) {
                return d['predictedYSmooth']
            }))
            quartile['estimate'] = Math.abs(d3.mean(filtData.map(function (d) {
                return d['predictedYSmooth']
            })) / d3.mean(filtData.map(function (d) {
                return d['predictedYSmooth']
            })))
            if (quartile['estimate'] >= 1) {
                returnDict['overEst'][cat] += appHelper.formatLabel(low) + "-" + appHelper.formatLabel(high) + ", "
            } else {
                returnDict['underEst'][cat] += appHelper.formatLabel(low) + "-" + appHelper.formatLabel(high) + ", "
            }
            statData['quartiles'].push(quartile)
        }
    }
    returnDict['lowest'] = d3.min(statData['quartiles'].map(function (d) {
        return d['averageImpact']
    }))
    returnDict['highest'] = d3.max(statData['quartiles'].map(function (d) {
        return d['averageImpact']
    }))
    returnDict['lowestGroup'] = statData['quartiles'].filter(function (d) {
        return d['averageImpact'] == returnDict['lowest']
    })[0]['group']
    returnDict['lowestGroup'] = returnDict['lowestGroup'].indexOf("Bin") >= 0 ? "for group " + returnDict['lowestGroup'].split("_")[1] : " for " + returnDict['lowestGroup']
    returnDict['highestGroup'] = statData['quartiles'].filter(function (d) {
        return d['averageImpact'] == returnDict['highest']
    })[0]['group']
    returnDict['highestGroup'] = returnDict['highestGroup'].indexOf("Bin") >= 0 ? "for group " + returnDict['highestGroup'].split("_")[1] : " for " + returnDict['highestGroup']
    returnDict['lowestQuartile'] = statData['quartiles'].filter(function (d) {
        return d['averageImpact'] == returnDict['lowest']
    })[0]['range']
    returnDict['highestQuartile'] = statData['quartiles'].filter(function (d) {
        return d['averageImpact'] == returnDict['highest']
    })[0]['range']
    returnDict['lowestRange'] = returnDict['lowestQuartile']
    returnDict['highestRange'] = returnDict['highestQuartile']
    returnDict['overAllAverageImpact'] = d3.mean(dataList.map(function (d) {
        return d['predictedYSmooth']
    }))
    returnDict['overAllMedianImpact'] = d3.median(dataList.map(function (d) {
        return d['predictedYSmooth']
    }))
    returnDict['diffMaxOverall'] = (returnDict['highest'] - returnDict['overAllMedianImpact'])
    returnDict['diffMaxMin'] = (returnDict['highest'] - returnDict['lowest'])
    return returnDict
    

}


//function to return the quartile data from the data provided for categorical variables

function createQuartileDataCategory(dataList, varX, cats, catlist) {
    var varRange = d3.extent(dataList.map(function (d) {
        return d[varX]
    }))
    var statData = {}
    var returnDict = {}
    returnDict['overEst'] = ''
    returnDict['underEst'] = ''
    statData['quartiles'] = []
    var diff = d3.quantile(varRange, 0.25) - d3.min(varRange)
    for (var i in catlist) {
        for (var c in cats) {
            var cat = cats[c]
            var catc = catlist[i]
            var quartile = {}
            quartile['quarter'] = catc
            quartile['group'] = cat
            var low = d3.min(varRange) + (i - 1) * diff
            var high = d3.min(varRange) + (i) * diff
            var filtData = dataList.filter(function (d) {
                return d[varX] == catc && d['groupByValue'] == cat
            })
            quartile['averageError'] = d3.mean(filtData.map(function (d) {
                return d['errPos'] + d['errNeg']
            }))
            quartile['MedianError'] = d3.median(filtData.map(function (d) {
                return d['errPos'] + d['errNeg']
            }))
            quartile['estimate'] = Math.abs(d3.mean(filtData.map(function (d) {
                return d['errPos']
            })) / d3.mean(filtData.map(function (d) {
                return d['errNeg']
            })))
            if (quartile['estimate'] >= 1) {
                returnDict['overEst'] += catc + " for " + cat + ", "
            } else {
                returnDict['underEst'] += catc + " for " + cat + ", "
            }
            statData['quartiles'].push(quartile)
        }
    }
    returnDict['lowest'] = d3.min(statData['quartiles'].map(function (d) {
        return d['averageError']
    }))
    returnDict['highest'] = d3.max(statData['quartiles'].map(function (d) {
        return d['averageError']
    }))
    returnDict['lowestGroup'] = statData['quartiles'].filter(function (d) {
        return d['averageError'] == returnDict['lowest']
    })[0]['group']
    returnDict['lowestGroup'] = returnDict['lowestGroup'].indexOf("Bin") >= 0 ? "for group " + returnDict['lowestGroup'].split("_")[1] : " for " + returnDict['lowestGroup']
    returnDict['highestGroup'] = statData['quartiles'].filter(function (d) {
        return d['averageError'] == returnDict['highest']
    })[0]['group']
    returnDict['highestGroup'] = returnDict['highestGroup'].indexOf("Bin") >= 0 ? "for group " + returnDict['highestGroup'].split("_")[1] : " for " + returnDict['highestGroup']
    returnDict['lowestQuartile'] = statData['quartiles'].filter(function (d) {
        return d['averageError'] == returnDict['lowest']
    })[0]['quarter']
    returnDict['highestQuartile'] = statData['quartiles'].filter(function (d) {
        return d['averageError'] == returnDict['highest']
    })[0]['quarter']

    returnDict['overAllAverageError'] = d3.mean(dataList.map(function (d) {
        return d['errPos'] + d['errNeg']
    }))
    returnDict['overAllMedianError'] = d3.median(dataList.map(function (d) {
        return d['errPos'] + d['errNeg']
    }))
    returnDict['diffMaxOverall'] = (returnDict['highest'] - returnDict['overAllMedianError'])
    returnDict['diffMaxMin'] = (returnDict['highest'] - returnDict['lowest'])
    return returnDict


}


//function to create the color dictionary by assigning a value to each group by variable
function createColorDict() {


    var categories = []

    for (var ind = 0; ind < AppData.length; ind++) {

        var chartRawData = AppData[ind]['Data'].filter(function (d) {
            return d['groupByVarName'] == groupBy
        })
        if (chartRawData.length > 0) {
            var varArray = chartRawData[0]
            var varX = Object.keys(varArray).filter(function (d) {
                return permKeys.indexOf(d) < 0
            })[0]
            colorDict[varX] = {}
            colorDict["heat"] = {}
            chartRawData.forEach(function (d) {
                if (categories.indexOf(d['groupByValue']) < 0) {
                    categories.push(d['groupByValue'])
                }
            })
            categories.sort(appHelper.sortAlphaNum)
            categories.forEach(function (c, i) {
                colorDict[varX][c] = colorList[i]
            })
        }
    }
    categories.forEach(function (c, i) {
        colorDict['heat'][c] = colorList[i]
    })
}


//Subsetting the data for the charts. Removing the treemap data as the code was built to handle only the error plots data
function prepareAppData() {
    AppData.filter(function (d, i) {
        if (d.Type == "Accuracy") {
            metaData['statData'] = d.Data
            metaData['ErrType'] = d.ErrType
            AppData.splice(i, 1)
        }
    })
    AppData.filter(function (d, i) {
        if (d.Type == "PercentileGroup") {
            metaData['percData'] = d.Data
            AppData.splice(i, 1)
        }
    })
    AppData.filter(function (d, i) {
        if (d.Type == "Percentile") {
            metaData['percGData'] = d.Data
            AppData.splice(i, 1)
        }
    })
}

//calling the required functions
prepareAppData()
prepareFilterData()
intializeFilterDropdown()
intializeTreeMap(AppData[0]['Data'][0]['groupByVarName'])


//Function that will loop across the data list for each variable and create the charts
function loopVariables(varInd) {

    //seeting up layout for each chart and intitalizing variables required
    var chartRawData = AppData[varInd]['Data'].filter(function (d) {
        return d['groupByVarName'] == groupBy && heatMapFilters.indexOf(d['groupByValue']) < 0
    })
    if (chartRawData.length == 0) {
        return null;
    }
    var changePar = 1
    var varArray = chartRawData[0]
    var varX = Object.keys(varArray).filter(function (d) {
        return permKeys.indexOf(d) < 0
    })[0]
    var margin = {
        top: 20,
        right: 15,
        bottom: 30,
        left: 60
    }
    var width = appHelper.getWidth() * 0.74 - margin.left - margin.right;
    var height = 300 - margin.top - margin.bottom;
    var chartLevel = mainApp.append("div").attr("class", "chart-lev").attr("id", "chlvl-" + varX.replace(/[^a-zA-Z]/g, "")).style("height", "430px")
    var main = chartLevel.append("div").attr("class", "app").attr("id", varX.replace(/[^a-zA-Z]/g, ""))
    var title = main.append('div').attr('class', 'Title')

    var filterContainer = main.append('div').attr('class', 'Filtercontainer')
    var total = main.append('div').attr('class', 'Total')
    var chartContainer = main.append('div').attr('class', 'Chartcont')
    var title = chartContainer.append('div').attr('class', 'Title')
    var legendGroup = chartContainer.append("div").attr("class", "legend").attr("width", width - 100).attr("height", height - 100)
    var narrative = chartLevel.append('div').attr('class', 'desc').style("height", "560px")
    var narrativeText = narrative
        .append("div")
        .attr('class', 'desc-text')

    var source = main.append('div').attr('class', 'Source')
    var attribution = main.append('div').attr('class', 'Attribution')


    var filterCategories = {}

    //filtering the data for that particular variable
    filterCategories[varX] = []

    //checking for categorical/continous variables
    if (AppData[varInd]['Type'] == "Continuous") {
        initializeLineChart()
    } else if (AppData[varInd]['Type'] == "Categorical") {
        initializeBarChart()
    }


    //function to initialize the line chart
    function initializeLineChart() {
        //creating the svg for the line chart
        svg = appHelper.createChart(chartContainer, width);
        appHelper.setTitle(title, "Error Distribution Graph: " + varX + " (Grouped by " + groupBy + ")");
        svg.selectAll('*').remove();
        svg.attr('width', width);
        //getting the list of the type variables present in the filtered data and sorting them
        var categories = []
        chartRawData.forEach(function (d) {
            if (categories.indexOf(d['groupByValue']) < 0) {
                categories.push(d['groupByValue'])
                filterCategories[varX].push(d['groupByValue'])
            }
        })
        //intializing few local variables

        var widthOffset = 140
        var legendPerRow = parseInt(width / widthOffset)
        var numOfRows = Math.ceil(categories.length / legendPerRow)

        //adding the legend for the percentile lines
        var percLegend = svg.append("g")
            .attr("transform", "translate(" + (100) + "," + (height + numOfRows * 30 + 35) + ")")
        var percLine = percLegend.append("line")
            .attr("x1", 0)
            .attr("y1", 0)
            .attr("x2", 20)
            .attr("y2", 0)
            .attr("stroke", "#75787B")
            .attr("stroke-width", "3px")
        var percLine = percLegend.append("text")
            .text("Percentile")
            .attr("x", 30)
            .attr("y", 5)

        //setting up the height based on the legend height
        svg.attr('height', height + numOfRows * 30);

        //intializing the scale for the axes 
        var x = d3.scaleLinear().range([margin.left, width])
        var y = d3.scaleLinear().range([height, 0])
        var z = d3.interpolateCool

        //Adding the line functions
        var line = d3.line()
            .x(function (d) {
                return x(d[varX])
            })
            .y(function (d) {
                return y(d['predictedYSmooth'])
            })
            .curve(d3.curveBasis)
        var area = d3.area()
            .x(function (d) {
                return x(d[varX])
            })
            .y0(function (d) {
                return y(d['predictedYSmooth'])
            })
            .y1(function (d) {
                return y(d['predictedYSmooth'] + (d['errPos'] == 'null' ? 0 : d['errPos']))
            })
            .curve(d3.curveBasis)
        var areaNeg = d3.area()
            .x(function (d) {
                return x(d[varX])
            })
            .y0(function (d) {
                return y(d['predictedYSmooth'] + (d['errNeg'] == 'null' ? 0 : d['errNeg']))
            })
            .y1(function (d) {
                return y(d['predictedYSmooth'])
            })
            .curve(d3.curveBasis)


        //creating elements for percentiles and percentile text
        var refLines = svg.append("g").attr("class", "percRefLines").attr("transform", "translate(10,0)")
        var refLinesText = svg.append("g").attr("class", "textRefLines").attr("transform", "translate(10,0)")

        //adding x axis element
        var xAxisElement = svg.append("g").attr("class", "xAxis Axes").attr("transform", "translate(10," + (height + numOfRows * 30 + 10) + ")")

        //adding yaxis element
        var yAxisElement = svg.append("g").attr("class", "yAxis Axes").attr("transform", "translate(60,30)")

        //adding group element for  paths adn circles
        var posPathGroup = svg.append("g").attr("class", "posErrorG").attr("transform", "translate(10," + (numOfRows * 30) + ")")
        var negPathGroup = svg.append("g").attr("class", "negErrorG").attr("transform", "translate(10," + (numOfRows * 30) + ")")
        var lineGroup = svg.append("g").attr("class", "lineG").attr("transform", "translate(10," + (numOfRows * 30) + ")")
        var circleGroup = svg.append("g").attr("class", "circleG").attr("transform", "translate(10," + (numOfRows * 30) + ")")
        var circleHighlightGroup = svg.append("g").attr("class", "circleHG").attr("transform", "translate(10," + (numOfRows * 30) + ")")

        //adding and formatting x labels
        var xLabel = svg.append("text").attr("class", "label").attr("transform", "translate(" + width / 2 + "," + (height + numOfRows * 30 + 40) + ")").style("text-anchor", "middle").text(varX)

        //adding an dformatting y-label
        var yLabel = svg.append("text").attr("class", "label").attr("transform", "translate(15," + (height / 2) + ")rotate(-90)").style("text-anchor", "end").text("Predicted-Quality")

        //adding element for on hover reference line
        var refLine = svg.append("line")
            .style("opacity", 0)
            .attr("x1", 0)
            .attr("y1", numOfRows * 30 - 10)
            .attr("x2", 0)
            .attr("y2", height + numOfRows * 30 + 10)
            .attr("stroke", "#75787B")

        //setting up a dummy inital domain for y-axis
        y.domain([-1, 1])

        //positioning and calling the y-axis
        var yAxis = d3.axisLeft().scale(y).tickSizeInner(-width + margin.left)
        var xAxis = d3.axisBottom().scale(x).tickSizeInner(-height - 10)
        yAxisElement.call(yAxis)
        xAxisElement.call(xAxis)


        //function to update the elements, this will help in updating in existing dom elements without creating new
        function drawChart(onlyResizeFlag) {
            posPathGroup.selectAll("*").remove()
            negPathGroup.selectAll("*").remove()
            lineGroup.selectAll("*").remove()
            circleGroup.selectAll("*").remove()
            circleHighlightGroup.selectAll("*").remove()


            //creating the quartile data
            var indata = createQuartileData(chartRawData, varX, filterCategories[varX])

            var html = "<ul style='list-style-type:disc' > <li>Across all categories, the model's error in predicting <b>" + yVar + " </b> is lowest from <b> [" + indata['lowestRange'] + "] </b> " + indata['lowestGroup'] + " and highest from <b> [" + indata['highestRange'] + "] </b> " + indata['highestGroup'] + "</li> <li> The error from <b> [" + indata['highestRange'] + "] </b>" + indata['highestGroup'] + " is  " + appHelper.formatLabel(indata['diffMaxOverall']) + " higher than the overall median error and " + appHelper.formatLabel(indata['diffMaxMin']) + " higher than the average error from <b> [" + indata['lowestRange'] + "] </b> " + indata['lowestGroup'] + "</li> <li> When the model makes prediction errors, there are times when those errors systematically lean in one direction </li><li>  The model tends to consistently make mistakes in the same direction when it miss-estimates " + varX 
            
            filterCategories[varX].forEach(function(d){
                if(indata['overEst'][d] != ""){
                html += ". Within the "+d+" group, the model consistently makes overestimates in the quartile range(s) <b>"+indata['overEst'][d].slice(0, indata['overEst'][d].length - 2)+"</b>"} 
                if(indata['underEst'][d] != ""){
                html += " while making underestimates in the quartile ranges <b>"+indata['underEst'][d].slice(0, indata['underEst'][d].length - 2)+"</b>"
                }
            })
            html += "</li></ul>"
            
            //updating the narrative text element based on the quartile data
            /*narrativeText.html("<ul style='list-style-type:disc' > <li>Across all categories, the model's error in predicting <b>" + yVar + " </b> is lowest from <b> [" + indata['lowestRange'] + "] </b> " + indata['lowestGroup'] + " and highest from <b> [" + indata['highestRange'] + "] </b> " + indata['highestGroup'] + "</li> <li> The error from <b> [" + indata['highestRange'] + "] </b>" + indata['highestGroup'] + " is  " + appHelper.formatLabel(indata['diffMaxOverall']) + " higher than the overall median error and " + appHelper.formatLabel(indata['diffMaxMin']) + " higher than the average error from <b> [" + indata['lowestRange'] + "] </b> " + indata['lowestGroup'] + "</li> <li> When the model makes prediction errors, there are times when those errors systematically lean in one direction </li><li>  The model tends to consistently make mistakes in the same direction when it miss-estimates " + varX + " between " + indata['overEst'].slice(0, indata['overEst'].length - 2) + " (overestimate) and " + indata['underEst'].slice(0, indata['underEst'].length - 2) + " (underestimate) </li></ul>")*/
            
            narrativeText.html(html)


            var newCategories = filterCategories[varX]
            var minVal = 100
            var maxVal = 0
            var chartData = chartRawData.filter(function (d) {
                return newCategories.indexOf(d['groupByValue']) >= 0
            })
            x.domain(d3.extent(chartData, function (d) {
                return d[varX]
            }))
            chartData.forEach(function (d) {
                var sumPos = (d['predictedYSmooth']) + (d['errPos'] == 'null' ? 0 : Math.abs(d['errPos']))
                var sumNeg = (d['predictedYSmooth']) + (d['errNeg'] == 'null' ? 0 : d['errNeg'])
                if (minVal > sumNeg) {
                    minVal = sumNeg
                }
                if (maxVal < sumPos) {
                    maxVal = sumPos
                }

            })

            //extracting the percentile data for each group by value
            var percDataList = []
            for (var p in percList) {
                var sum = 0
                var sumProp = 0

                for (var c in filterCategories[varX]) {
                    var cat = filterCategories[varX][c]

                    var perc = filterCategories[varX].length== categories.length?getXthPercentaileGlobal(chartRawData, percList[p], cat, varX):getXthPercentaile(chartRawData, percList[p], cat, varX)
                    var prop = metaData['proportion'].filter(function (d) {
                        return d.name == cat
                    })[0]['perc'] / 100
                    sum += perc * prop
                    sumProp += prop
                }
                percDataList.push({
                    "percentile": percList[p],
                    "value": appHelper.formatLabel(sum / sumProp)
                })

            }


            //sorting the data from maximum to minimum
            chartData = chartData.sort(function (a, b) {
                return a[varX] - b[varX]
            })

            //setting up the domain for y-axis based on the stacked values
            y.domain([minVal, maxVal])
            yAxisElement.transition().duration(1000).call(yAxis)

            //updating the percentile line positions based on the data check box option
            if (percFlag & newCategories.length != 0) {
                var refs = refLines.selectAll("line")
                    .data(percDataList).style("opacity",1)
                refs.exit().remove()
                refs.enter().append("line")
                    .style("opacity", 1)
                    .attr("x1", function (d) {
                        return x(d['value'])
                    })
                    .attr("y1", numOfRows * 30)
                    .attr("x2", function (d) {
                        return x(d['value'])
                    })
                    .attr("y2", height + numOfRows * 30 + 10)
                    .attr("stroke", "#75787B")
                    .attr("stroke-width", "2px")

                refs.transition().attr("x1", function (d) {
                        return x(d['value'])
                    })
                    .attr("y1", numOfRows * 30)
                    .attr("x2", function (d) {
                        return x(d['value'])
                    })
                    .attr("y2", height + numOfRows * 30 + 10)
                    .attr("stroke", "#75787B")
                    .attr("stroke-width", "2px")
                var refsText = refLinesText.selectAll("text")
                    .data(percDataList).style("opacity",1)
                refsText.exit().remove()
                refsText.enter().append("text")
                    .text(function (d) {
                        return d['percentile']
                    })
                    .attr('transform', function (d) {
                        return "translate(" + (x(d['value']) - 5) + ",25)"
                    })

                refsText.transition()
                    .text(function (d) {
                        return d['percentile']
                    })
                    .attr('transform', function (d) {
                        return "translate(" + (x(d['value']) - 5) + ",25)"
                    })

            }
            if(newCategories.length == 0){
                refLines.selectAll("line").style("opacity",0)
                refLinesText.selectAll("text").style("opacity",0)
            }

            //updating the patrh line
            var linePaths = lineGroup.selectAll("path")
                .data(newCategories)
                .enter().append("path")
                .attr("fill", "none")
                .attr("class", function (d) {
                    return "errorline " + d
                })
                .attr("stroke", function (d) {
                    return colorDict[varX][d]
                })
                .attr("stroke-width", "2px")
                .datum(function (d) {

                    var filteredData = []
                    chartData.map(function (k) {
                        if (k['groupByValue'] == d) {
                            filteredData.push(k)
                        }
                    });
                    return filteredData
                })
                .attr("d", line)
            /*var circles = circleGroup.selectAll("g")
                .data(newCategories)
                .enter().append("g")
                .attr("class", function (d) {
                    return "circleGroup " + d
                })
            var circle = circles.selectAll("circle")
                .data(function (d) {
                    var filteredData = []
                    var filteredData = []
                    chartData.map(function (k) {
                        if (k['groupByValue'] == d) {
                            filteredData.push(k)
                        }
                    });
                    return filteredData
                }).enter().append("circle")
                .attr("class", function (d) {
                    return "circle " + d['groupByValue']
                })
                .attr("id", "circle" + varX)
                .attr("cx", function (d) {
                    return x(d[varX])
                })
                .attr("cy", function (d) {
                    return y(d['predictedYSmooth'])
                })
                .attr("r", function (d) {
                    return d["highlight"] == "N" ? 0 : 3
                })
                .attr("fill", function (d) {
                    return colorDict[varX][d['groupByValue']]
                })
            var circlesHighlight = circleHighlightGroup.selectAll("g")
                .data(newCategories)
                .enter().append("g")
                .attr("class", function (d) {
                    return "circleHGroup " + d
                })
            var circleHighlight = circlesHighlight.selectAll("circle")
                .data(function (d) {
                    var filteredData = []
                    chartData.map(function (k) {
                        if (k['groupByValue'] == d) {
                            filteredData.push(k)
                        }
                    });
                    return filteredData
                }).enter().append("circle")
                .attr("class", function (d) {
                    return "circleH " + d['groupByValue']
                })
                .attr("id", "circle " + varX)
                .attr("cx", function (d) {
                    return x(d[varX])
                })
                .attr("cy", function (d) {
                    return y(d['predictedYSmooth'])
                })
                .attr("r", function (d) {
                    return d["highlight"] == "N" ? 0 : 6
                })
                .attr("fill", function (d) {
                    return colorDict[varX][d['groupByValue']]
                })
                .style("opacity", 0.4)*/

            var pospaths = posPathGroup.selectAll("path")
                .data(newCategories)
                .enter().append("path")
                .attr("class", function (d) {
                    return "errorline posError " + d
                })
                .attr("fill", "grey")
                .style("opacity", 0.4)
                .datum(function (d) {
                    var filteredData = []
                    chartData.map(function (k) {
                        if (k['groupByValue'] == d) {
                            filteredData.push(k)
                        }

                    });
                    return filteredData
                })
                .attr("id", "line" + varX.replace(/[^a-zA-Z]/g, ""))
                .attr("d", area)
            var negpaths = negPathGroup.selectAll("path")
                .data(newCategories)
                .enter().append("path")
                .attr("fill", "grey")
                .attr("class", function (d) {
                    return "errorline negError " + d
                })
                .style("opacity", 0.4)
                .datum(function (d) {
                    var filteredData = []
                    chartData.map(function (k) {
                        if (k['groupByValue'] == d) {
                            filteredData.push(k)
                        }
                    });
                    return filteredData
                })
                .attr("id", "line" + varX.replace(/[^a-zA-Z]/g, ""))
                .attr("d", areaNeg)

            //adding event handlers for the elements
            d3.select("#" + varX.replace(/[^a-zA-Z]/g, "")).selectAll(".errorline")
                .on("mouseenter", mouseenter)
                .on("mouseout", mouseout)
                .on("mousemove", mousemove)

            //updating the tick elements
            xAxisElement.transition().duration(1000).call(xAxis)
            xAxisElement.selectAll("line").attr("opacity", 0.25).style("stroke-dasharray", ("3, 3"))
            yAxisElement.selectAll("line").attr("opacity", 0.25).style("stroke-dasharray", ("3, 3"))

            //adding legend buttons in the top
            var legend = legendGroup.selectAll("button")
                .data(categories.sort(appHelper.sortAlphaNum))
                .enter().append("button")
                .attr("class", function (d) {
                    return "legend-btn " + d
                })
                .style("background-color", function (d) {
                    return colorDict[varX][d]
                })
                .style("cursor", "pointer")
                .on("click", legendClick)
                .on("mouseover", legendMouseOver)
                .on("mouseout", legendMouseOut)
                .html(function (d) {
                    return d
                })

            //event listeners for the legend buttons
            function legendMouseOver(d, i) {
                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                svgCurrent.selectAll(".errorline").style("opacity", 0.25)
                svgCurrent.selectAll("." + d).style("opacity", 1)

            }

            function legendMouseOut(d, i) {
                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                svgCurrent.selectAll(".errorline").style("opacity", 1)
                svgCurrent.selectAll(".posError").style("opacity", 0.4)
                svgCurrent.selectAll(".negError").style("opacity", 0.4)
                svgCurrent.selectAll(".circleH").style("opacity", 0.4)
            }
            //functions to show mosue over values for the area charts
            function mousemove(d, i) {
                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                var className = d3.select(this).attr('class').split(" ")[2]
                var closestXValue = getClosestxValue()

                refLine.transition().style("opacity", 1).attr("x1", x(closestXValue)).attr("x2", x(closestXValue)).attr("transform", "translate(10,0)")
                setTimeout(function () {
                    showingTooltip = true;
                }, 300)
                var filt = chartRawData.filter(function (d) {
                    return d[varX] == closestXValue && d['groupByValue'] == className
                })[0]
                tooltip.attr('class', "colorClass" + varInd + i);
                tooltip.transition().duration(200).delay(100).style("opacity", 1);
                tooltip.html("At <span>" + appHelper.formatLabel(filt[varX]) + ", " + varX + " </span>has a Predicted-Quality of <span>" + appHelper.formatLabel(filt["predictedYSmooth"]) + " </span> with a positive error of <span>" + (filt['errPos'] != 'null' ? appHelper.formatLabel(filt['errPos']) : 0) + "</span> and a negative error of <span> " + (filt['errNeg'] != 'null' ? appHelper.formatLabel(filt['errNeg']) : 0) + "</span>")
                    .style("left", function () {
                        var tooltipWidth = this.getBoundingClientRect().width;
                        var currentMouseX = d3.event.pageX;
                        if (width > tooltipWidth + currentMouseX + 25) {
                            return (d3.event.pageX + 5) + "px";
                        } else {
                            if (d3.event.pageX - tooltipWidth - 5 < 10) {
                                return "10px";
                            }
                            return (d3.event.pageX - tooltipWidth - 5) + "px";
                        }
                    })
                    .style("top", (d3.event.pageY + 20) + "px")
                tooltip.style("color", colorDict[varX][filt['groupByValue']])

                svgCurrent.selectAll(".errorline").style("opacity", 0.1)
                svgCurrent.selectAll("circle").style("opacity", 0.1)
                svgCurrent.selectAll("." + className).style("opacity", 1).attr("cursor", "pointer")
                svgCurrent.selectAll(".circleH").filter("." + className).style("opacity", 0.4)
            }

            function mouseout(d) {

                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                var className = d3.select(this).attr('class').split(" ")[1]
                svgCurrent.selectAll("path").style("opacity", 1)
                svgCurrent.selectAll(".negError").style("opacity", 0.25)
                svgCurrent.selectAll(".posError").style("opacity", 0.25)
                svgCurrent.selectAll("circle").style("opacity", 1)
                svgCurrent.selectAll(".circleH").style("opacity", 0.4)
                tooltip.transition().duration(200).delay(100).style("opacity", 0)

                refLine.transition().style("opacity", 0)
            }

            function mouseenter(d, i) {
                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                var className = d3.select(this).attr('class').split(" ")[2]
                var closestXValue = getClosestxValue()

                refLine.transition().style("opacity", 1).attr("x1", x(closestXValue)).attr("x2", x(closestXValue)).attr("transform", "translate(10,0)")
                setTimeout(function () {
                    showingTooltip = true;
                }, 300)
                var filt = chartRawData.filter(function (d) {
                    return d[varX] == closestXValue && d['groupByValue'] == className
                })[0]
                tooltip.attr('class', "colorClass" + varInd + i);
                tooltip.transition().duration(200).delay(100).style("opacity", 1);
                tooltip.html("At <span>" + appHelper.formatLabel(filt[varX]) + ", " + varX + " </span>has a Predicted-Quality of <span>" + appHelper.formatLabel(filt["predictedYSmooth"]) + " </span> with a positive error of <span>" + (filt['errPos'] != 'null' ? appHelper.formatLabel(filt['errPos']) : 0) + "</span> and a negative error of <span> " + (filt['errNeg'] != 'null' ? appHelper.formatLabel(filt['errNeg']) : 0) + "</span>")
                    .style("left", function () {
                        var tooltipWidth = this.getBoundingClientRect().width;
                        var currentMouseX = d3.event.pageX;
                        if (width > tooltipWidth + currentMouseX + 25) {
                            return (d3.event.pageX + 5) + "px";
                        } else {
                            if (d3.event.pageX - tooltipWidth - 5 < 10) {
                                return "10px";
                            }
                            return (d3.event.pageX - tooltipWidth - 5) + "px";
                        }
                    })
                    .style("top", (d3.event.pageY + 20) + "px")
                tooltip.style("color", colorDict[varX][filt['groupByValue']])

                svgCurrent.selectAll(".errorline").style("opacity", 0.1)
                svgCurrent.selectAll("circle").style("opacity", 0.1)
                svgCurrent.selectAll("." + className).style("opacity", 1).attr("cursor", "pointer")
                svgCurrent.selectAll(".circleH").filter("." + className).style("opacity", 0.4)
            }
            //Legend click event handlers
            function legendClick(d, i) {
                if (filterCategories[varX].indexOf(d) >= 0) {
                    filterCategories[varX].splice(filterCategories[varX].indexOf(d), 1)
                    drawChart(false)
                    d3.select(this).style("opacity", 0.5).style("background-color", "#A7A8AA")

                } else {
                    filterCategories[varX].push(d)
                    drawChart(false)
                    d3.select(this).style("opacity", 1).style("background-color", colorDict[varX][d])
                }
            }

        }
        //helper function to set the reference value at the mouse over position
        function getClosestxValue() {
            var xPos = x.invert(d3.event.pageX - appHelper.getWidth() * 0.02)
            var xValues = chartRawData.map(function (d) {
                return d[varX]
            })
            var xClosest = getClosest(xPos, xValues)
            return xClosest
        }
        //getting the closest value from the list 
        function getClosest(num, arr) {
            var curr = arr[0];
            var diff = Math.abs(num - curr);
            for (var val = 0; val < arr.length; val++) {
                var newdiff = Math.abs(num - arr[val]);
                if (newdiff < diff) {
                    diff = newdiff;
                    curr = arr[val];
                }
            }
            return curr;
        }


        //Initial setup
        drawChart(false);

    }

    function initializeBarChart() {

        //function to extract the list of categories in the data
        function getCategories(data) {
            var categories = []
            data.forEach(function (d) {
                if (categories.indexOf(d[varX]) < 0) {
                    categories.push(d[varX])
                }
            })
            return categories
        }
        var maxHeight = height+30
        //extarcting the group by varaibles
        var categoryData = getCategories(chartRawData)
        // creating the svg element for bar charts
        svg = appHelper.createChart(chartContainer, width);
        //settign up title for the chart
        appHelper.setTitle(title, "Error Distribution Graph: " + varX + " (Grouped by "+groupBy +") ");

        //setting up the layout for the chart
        svg.selectAll('*').remove();
        svg.attr('width', width);
        var categories = []

        chartRawData.forEach(function (d) {
            if (categories.indexOf(d['groupByValue']) < 0) {
                categories.push(d['groupByValue'])
                filterCategories[varX].push(d['groupByValue'])
            }
        })

        var stackColumns = ["predictedYSmooth", "errPos", "errNeg"]
        var widthOffset = 140
        var legendPerRow = parseInt(width / widthOffset)
        var numOfRows = Math.ceil(categories.length / legendPerRow)
        svg.attr('height', 450 + numOfRows * 30);
        var stack = d3.stack()

        //intializing the axes
        var x = d3.scaleBand().range([margin.left, width]).paddingInner(0.25)
        var x1 = d3.scaleBand().padding(0.25)
        var y = d3.scaleLinear().range([height - 30, 0])
        svg.append("g").attr("class", "xAxis").attr("transform", "translate(10," + (height + numOfRows * 30 + 20) + ")")

        //creating the axes elements
        var yAxisElement = svg.append("g").attr("class", "yAxis").attr("transform", "translate(60,70)")
        var barGroup = svg.append("g").attr("transform", "translate(0,70)")
        var xLabel = svg.append("text").attr("class", "label").attr("transform", "translate(" + width / 2 + "," + (height + numOfRows * 30 + 60) + ")").style("text-anchor", "middle").text(varX)
        var yLabel = svg.append("text").attr("class", "label").attr("transform", "translate(15," + (height / 2) + ")rotate(-90)").style("text-anchor", "end").text("Predicted-Quality")
        var refLine = svg.append("line")
            .style("opacity", 0)
            .attr("x1", 0)
            .attr("y1", numOfRows * 30 + 20)
            .attr("x2", 0)
            .attr("y2", height + numOfRows * 30)
            .attr("stroke", "#75787B")

        //setting up dummy domain value for y-axis and calling the y-axiss
        y.domain([-10, 10])

        var yAxis = d3.axisLeft().scale(y).tickSizeInner(-width + margin.left)
        yAxisElement.call(yAxis)
        var dataGroup = svg.append("g")

        //function to update the elements, this will help in updating in existing dom elements without creating new
        function drawChart() {

            //removing the elements before initialize
            barGroup.selectAll("*").remove()
            dataGroup.selectAll("*").remove()

            //adding the lines 
            var posLineGroup = dataGroup.append("g")
                .attr("class", "posLineGroup")
                .attr("transform", "translate(0,70)")
            var dashedLineGroup = dataGroup.append("g")
                .attr("class", "dashedLineGroup")
                .attr("transform", "translate(0,70)")
            var circleGroup = dataGroup.append("g")
                .attr("class", "circleGroup")
                .attr("transform", "translate(0,70)")

            var negLineGroup = dataGroup.append("g")
                .attr("class", "negLineGroup")
                .attr("transform", "translate(0,70)")
            var newCategories = filterCategories[varX].sort(appHelper.sortAlphaNum)
            var minVal = 1000
            var maxVal = 0


            //updating the bar elements on filtering
            var barData = chartRawData.filter(function (d) {
                return newCategories.indexOf(d['groupByValue']) >= 0
            })
            barData.forEach(function (d) {
                var posSum = Math.abs(d['predictedYSmooth']) + Math.abs(d['errPos'])
                var negSum = Math.abs(d['predictedYSmooth']) - Math.abs(d['errNeg'])
                minVal = d3.min([minVal, posSum, negSum])
                maxVal = d3.max([maxVal, posSum, negSum])

            })
            //getting the range of x value
            var chartRange = [minVal > 0 ? 0.9 * minVal : 1.1 * minVal, maxVal < 0 ? maxVal * 0.9 : maxVal * 1.1]

            //setting up the domain values for x and y
            x.domain(categoryData)
            x1.domain(newCategories).rangeRound([0, x.bandwidth()])
            y.domain(chartRange)

            //getting the quartile data
            var indata = createQuartileDataCategory(barData, varX, filterCategories[varX], categoryData)

            //updating the narrative text
            narrativeText.html("<ul style='list-style-type:disc' > <li>Across all categories, the model's error in predicting <b>" + yVar + " </b> is lowest for <b> " + indata['lowestQuartile'] + " </b> " + indata['lowestGroup'] + " and highest for <b> " + indata['highestQuartile'] + " </b> " + indata['highestGroup'] + "</li> <li> The error for <b>" + indata['highestQuartile'] + " </b> " + indata['highestGroup'] + " is  " + appHelper.formatLabel(indata['diffMaxOverall']) + " higher than the overall median error and " + appHelper.formatLabel(indata['diffMaxMin']) + " higher than the average error from <b>" + indata['lowestQuartile'] + "</b> " + indata['lowestGroup'] + "</li> <li> When the model makes prediction errors, there are times when those errors systematically lean in one direction </li><li>  The model tends to consistently make mistakes in the same direction when it miss-estimates " + varX + " between " + indata['overEst'].slice(0, indata['overEst'].length - 2) + " (overestimate) and " + indata['underEst'].slice(0, indata['underEst'].length - 2) + " (underestimate) </li></ul>")

            //updating the axes elements
            var xAxis = d3.axisBottom().scale(x)
            yAxisElement.transition().duration(1000).call(yAxis)
            svg.select(".xAxis").call(xAxis).attr("transform", "translate(0," + (height + 50) + ")")
            
            svg.select(".xAxis")
                .selectAll("text")
                .attr("x", function (d) {
                    return 0;
                })
                .call(wrap, x.bandwidth()*0.9)

            //updating the elements as per the data
            var dashedLineCatGroup = dashedLineGroup.selectAll("g")
                .data(categoryData)
                .enter().append("g")
                .selectAll("line")
                .data(function (d) {
                    return barData.filter(
                        function (p) {
                            return p[varX] == d
                        })
                })
                .enter().append("line")
                .attr("x1", function (d) {
                    return x(d[varX]) + x1(d['groupByValue']) + x1.bandwidth() / 2
                })
                .attr("y1", function (d) {
                    return y(d['predictedYSmooth'] + d['errNeg'])
                }).attr("x2", function (d) {
                    return x(d[varX]) + x1(d['groupByValue']) + x1.bandwidth() / 2
                })
                .attr("y2", function (d) {
                    return y(d['predictedYSmooth'] + d['errPos'])
                })
                .attr("stroke-width", 2)
                .style("stroke-dasharray", ("4, 4"))
                .attr("opacity", 1)
                .attr("stroke", "#75787B")


            var circleCatGroup = circleGroup.selectAll("g")
                .data(categoryData)
                .enter().append("g")
                .attr("class", function (d) {
                    return "circleGroup " + d
                })
                .selectAll("circle")
                .data(function (d) {
                    return barData.filter(
                        function (p) {
                            return p[varX] == d
                        })
                })
                .enter().append("circle")
                .attr("class", function (d) {
                    return "circle " + d['groupByValue']
                })
                .attr("r", 5)
                .attr("cx", function (d) {
                    return x(d[varX]) + x1(d['groupByValue']) + x1.bandwidth() / 2
                })
                .attr("cy", function (d) {
                    return y(d['predictedYSmooth'])
                })
                .attr("fill", function (d) {
                    return colorDict[varX][d['groupByValue']]
                })



            var posLineCatGroup = posLineGroup.selectAll("g")
                .data(categoryData)
                .enter().append("g")
                .selectAll("rect")
                .data(function (d) {
                    return barData.filter(
                        function (p) {
                            return p[varX] == d
                        })
                })
                .enter().append("rect")
                .attr("x", function (d) {
                    return x(d[varX]) + x1(d['groupByValue']) + x1.bandwidth() / 4
                })
                .attr("y", function (d) {
                    return y(d['predictedYSmooth'] + d['errPos'])
                }).attr("width", function (d) {
                    return x1.bandwidth() / 2
                })
                .attr("height", function (d) {
                    return 1
                })
                .attr("stroke-width", 3)
                .attr("opacity", 1)
                .attr("fill", "#75787B")



            var negLineCatGroup = negLineGroup.selectAll("g")
                .data(categoryData)
                .enter().append("g")
                .selectAll("rect")
                .data(function (d) {
                    return barData.filter(
                        function (p) {
                            return p[varX] == d && filterCategories[varX].indexOf(p['groupByValue']) >= 0
                        })
                })
                .enter().append("rect")
                .attr("x", function (d) {
                    return x(d[varX]) + x1(d['groupByValue']) + x1.bandwidth() / 4
                })
                .attr("y", function (d) {
                    return y(d['predictedYSmooth'] + d['errNeg'])
                }).attr("width", function (d) {
                    return x1.bandwidth() / 2
                })
                .attr("height", function (d) {
                    return 1
                })
                .attr("stroke-width", 3)
                .attr("opacity", 1)
                .attr("fill", "#75787B")


            /*categoryData.forEach(function (cat, catInd) {
                stackColumns.forEach(function (stack, stackInd) {
                    var key = ""
                    var chartData = chartRawData.filter(function (d) {
                        return d[varX] == cat
                    })
                    var levelOne = barGroup.append("g").attr("class", "category " + cat).attr("transform", "translate(" + (x(cat)) + ",0)")



                    levelOne.selectAll("g")
                        .data(newCategories)
                        .enter().append("g")
                        .attr("class", function (d) {
                            return "rectGroup " + d
                        })
                        .attr("transform", function (d, i) {
                            return "translate(" + (x1(d)) + ",0)"
                        })
                        .selectAll("rect")
                        .data(function (d, i) {
                            var filteredData = []
                            barData.map(function (k) {
                                if (k['groupByValue'] == d && k[varX] == cat) {
                                    filteredData.push(k)
                                }
                            });
                            return filteredData
                        })
                        .enter().append("rect")
                        .attr("class", function (d) {
                            return "rect " + d['groupByValue']
                        })
                        .attr("x", 0)
                        .attr("y", function (d) {
                            if (stack == 'errPos') {
                                return y(d['predictedYSmooth'] + d[stack])
                            } else {
                                return y(d['predictedYSmooth'])
                            }
                        })
                        .attr("height", function (d) {
                            if (stack == 'predictedYSmooth') {
                                return y(0) - y(d[stack])
                            } else if (stack == 'errPos') {
                                return y(0) - y(d['errPos'])
                            } else {
                                return y(0) - y(Math.abs(d['errNeg']))
                            }
                        })
                        .attr("width", function (d) {
                            if (stack == 'predictedYSmooth') {
                                return x1.bandwidth()
                            } else {
                                return x1.bandwidth()
                            }
                        })
                        .attr("fill", function (d, i) {
                            if (stack == 'predictedYSmooth') {
                                return colorDict[varX][d['groupByValue']]
                            } else {
                                return "grey"
                            }
                        })
                        .attr("transform", function (d) {
                            if (stack == 'predictedYSmooth') {
                                return "translate(0,0)"
                            }
                            return "translate(" + x1.bandwidth() * 0 + ",0)"
                        })
                        .style("opacity", function (d) {
                            if (stack == 'predictedYSmooth') {
                                return 1
                            }
                            return 0.5
                        })



                })
            })*/


            svg.selectAll("circle")
                .on("mouseenter", mouseover)
                .on("mouseout", mouseout)
                .on("mousemove", mousemove)
                .attr("cursor", "pointer")

            //positining the axes elements
            svg.select(".xAxis").selectAll("line").attr("opacity", 0.15).style("stroke-dasharray", ("4, 4"))
            svg.select(".yAxis").selectAll("line").attr("opacity", 0.15).style("stroke-dasharray", ("4, 4"))
            svg.select(".xAxis")
                .selectAll("text")
                .attr("x", function (d) {
                    return 0;
                })

            //adding the legend elements
            var legend = legendGroup.selectAll("button")
                .data(categories.sort(appHelper.sortAlphaNum))
                .enter().append("button")
                .attr("class", "legend-btn")
                .style("background-color", function (d) {
                    return colorDict[varX][d]
                })
                .style("cursor", "pointer")
                .on("click", legendClick)
                .on("mouseover", legendMouseOver)
                .on("mouseout", legendMouseOut)
                .html(function (d) {
                    return d
                })

            //adding the mouse elements
            function mouseover(d, i) {
                tooltip.transition().duration(200).delay(100).style("opacity", 1);
                tooltip.html("<span>" + varX + "</span> has a Predicted-Quality of <span>" + appHelper.formatLabel(d["predictedYSmooth"]) + "</span> for category <span>" + d['groupByValue'] + "</span> with errors in ranging, on average, from <span>" + appHelper.formatLabel(d['errNeg']) + "</span> to <span>" + appHelper.formatLabel(d["errPos"]) + "</span>")
                    .style("left", function () {
                        var tooltipWidth = this.getBoundingClientRect().width;
                        var currentMouseX = d3.event.pageX;
                        if (width > tooltipWidth + currentMouseX + 25) {
                            return (d3.event.pageX + 5) + "px";
                        } else {
                            if (d3.event.pageX - tooltipWidth - 5 < 10) {
                                return "10px";
                            }
                            return (d3.event.pageX - tooltipWidth - 5) + "px";
                        }
                    })
                    .style("top", (d3.event.pageY + 15) + "px")
                tooltip.style("color", colorDict[varX][d['groupByValue']])
            }

            function mousemove(d, i) {
                tooltip.transition().duration(200).delay(100).style("opacity", 1);
                tooltip.html("<span>" + varX + "</span> has a Predicted-Quality of <span>" + appHelper.formatLabel(d["predictedYSmooth"]) + "</span> for category <span>" + d['groupByValue'] + "</span> with errors in ranging, on average, from <span>" + appHelper.formatLabel(d['errNeg']) + "</span> to <span>" + appHelper.formatLabel(d["errPos"]) + "</span>")
                    .style("left", function () {
                        var tooltipWidth = this.getBoundingClientRect().width;
                        var currentMouseX = d3.event.pageX;
                        if (width > tooltipWidth + currentMouseX + 25) {
                            return (d3.event.pageX + 5) + "px";
                        } else {
                            if (d3.event.pageX - tooltipWidth - 5 < 10) {
                                return "10px";
                            }
                            return (d3.event.pageX - tooltipWidth - 5) + "px";
                        }
                    })
                    .style("top", (d3.event.pageY + 15) + "px")
                tooltip.style("color", colorDict[varX][d['groupByValue']])
            }

            //addign events to the legend buttons
            function legendMouseOver(d, i) {
                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                svgCurrent.selectAll("." + d).style("stroke", "#000").style("stroke-width", 2)
            }

            function legendMouseOut(d, i) {

                var svgCurrent = d3.select("#" + varX.replace(/[^a-zA-Z]/g, ""))
                svgCurrent.selectAll("." + d).style("stroke-width", 0)
            }

            function mouseout(d, i) {
                tooltip.transition().style("opacity", 0);
                showingTooltip = false;
                refLine.transition().style("opacity", 0)

                d3.select(this).attr("stroke-width", "0px")
            }

            function legendClick(d, i) {
                if (filterCategories[varX].indexOf(d) >= 0) {
                    filterCategories[varX].splice(filterCategories[varX].indexOf(d), 1)
                    drawChart()
                    d3.select(this).style("opacity", 0.5).style("background-color", "#A7A8AA")

                } else {
                    filterCategories[varX].push(d)
                    drawChart()
                    d3.select(this).style("opacity", 1).style("background-color", colorDict[varX][d])
                }
            }
            function wrap(text, width) {
                text.each(function () {
                    var text = d3.select(this),
                        words = text.text().split(/\s+/).reverse(),
                        word,
                        line = [],
                        lineNumber = 0,
                        lineHeight = 1.1, // ems
                        y = text.attr("y"),
                        dy = parseFloat(text.attr("dy")),
                        tspan = text.text(null).append("tspan").attr("x", 0).attr("y", y).attr("dy", dy + "em");
                    while (word = words.pop()) {
                        line.push(word);
                        tspan.text(line.join(" "));
                        if (tspan.node().getComputedTextLength() > width) {
                            line.pop();
                            tspan.text(line.join(" "));
                            line = [word];
                            tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
                        }
                    }
                    
                    //seeting up te height of the chart based on the number of lines in the wrap text
                    maxHeight = d3.max([450+lineNumber*20,maxHeight])
                    d3.select("#chlvl-"+varX.replace(/[^a-zA-Z]/g, "")).style("height",maxHeight+"px")
                });
            }
        }
        drawChart()
    }


}


//Event handlers
window.onresize = function () {
    d3.select(".Chartcontainer").selectAll("*").remove()
    width = appHelper.getWidth();

    intializeTreeMap(AppData[0]['Data'][0]['groupByVarName'])
    for (var ind = 0; ind < AppData.length; ind++) {
        loopVariables(ind)
    }

}
