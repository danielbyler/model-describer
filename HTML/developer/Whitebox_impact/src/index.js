//Importing helper functions
import AppHelper from './AppHelper.js';
//Importing Data
import { AppData } from './AppData.js';
//SASS Import to DOM at runtime
require('./styles/index.scss');
//Creatign helper functions object
var appHelper = new AppHelper();

//Creating the layout for all the elements
var main = d3.select('#App');


//Main Heading
var heading = main.append("div").attr('class', "heading").append("a").attr("href", "https://github.com/DataScienceSquad/WhiteBox_Production").html("White Box - Impact By Variable")
//Heat map and the type dropdown
var summary = main.append("div").attr("class", "summary")
var heatMapContainer = summary.append('div').attr('class', 'heatMapContainer')
var filterContainer = summary.append('div').attr('class', 'filt-cont')


//Main chart container
var mainApp = main.append('div').attr('class', 'Chartcontainer')
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
var colorList =["#86BC25", "#00A3E0", "#00ABAB", "#C4D600",  "#2C5234", "#9DD4CF", "#004F59", "#62B5E5", "#43B02A", "#0076A8","#9DD4CF", "#012169"]
var margin = {
    top: 20,
    right: 15,
    bottom: 60,
    left: 60
}
var permKeys = ['predictedYSmooth','groupByVarName','groupByValue','highlight','errNeg','errPos']
var percFlag = true;
var yVar = 'Quality'
var showingTooltip = false;
var percList = [10,50,90]
var heatMapData = {}
var colorDict = {}
var width, svg, svgHeat, globalHeightOffset, percentScale, responses;
var groupBy = ''
var heatMapFilters = []
var metaData = {}


//Creating svg container for Heat map
svgHeat = appHelper.createChart(heatMapContainer, width)


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
function togglePercentile(){
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
                                    .attr("class","perc-check")
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
        .html("Group <b>" + statEach['groupByValue'] + " </b> is " + percEach['perc'] + "% of the data and has an average error of <b>" + appHelper.formatLabel(statEach['MSE']) + "</b>")
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
        .html("Group <b>" + statEach['groupByValue'] + " </b> is " + appHelper.formatLabel(percEach['perc']) + "% of the data and has an average error of <b>" + appHelper.formatLabel(statEach['MSE']) + "</b>")
}

//Function to hide the tooltip on mouse out of treemap
function hideToolTipTreeMap(d, i) {
    tooltip.style("opacity", 0)
}


//Function to create treemap
function intializeTreeMap(type) {

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
    metaData['proportion'] = root.data.children
    var sumProportion = 0
    for (var p in metaData['proportion']) {
        sumProportion += metaData['proportion'][p]['size']
    }
    for (var p in metaData['proportion']) {
        metaData['proportion'][p]['perc'] = metaData['proportion'][p]['size'] * 100 / sumProportion
    }
    treemap(root);

    /*var title = svgHeat.selectAll(".title")
        .data(["Proportion by each variable"])
        .enter().append("text")
        .attr("y", "5px")
        .attr("class", "title")
        .text(function (d) {
            return d
        })
        .style("font-size", "16px")*/
    var cellGroup = svgHeat.append("g").attr("class", "cell-group").attr("transform", "translate(0,20)")
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
                return "Average Error: " + appHelper.formatLabel(statEach['MSE']);
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
    //console.log(sample)
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
        .entries(filteredSample)
    
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
//Function to get the xth percentile of the data given
function getXthPercentaile(dataList, x, cat, varX) {
    
    var varDict = metaData['percData'].filter(function(d){
        return d.variable == varX
    })[0]
    
    var typeDict = varDict['percentileList'].filter(function(d){
        return d.groupByVar == cat
    })[0]
    
    var percDict = typeDict['percentileValues'].filter(function(d){
        return d.percentiles == x+"%"
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
    if(cats.length==0) return 0;
    var varRange = d3.extent(dataList.map(function (d) {
        return d[varX]
    }))
    var statData = {}
    var returnDict = {}
    returnDict['overEst'] = ''
    returnDict['underEst'] = ''
    statData['quartiles'] = []
    var diff = d3.quantile(varRange, 0.25) - d3.min(varRange)
    
    for (var i = 1; i <= 4; i++) {
        for (var c in cats) {

            var cat = cats[c]
            var quartile = {}
            quartile['quarter'] = i
            quartile['group'] = cat
            var low = getXthPercentaile(dataList,(i-1)*25,cat,varX)
            var high = getXthPercentaile(dataList,(i)*25,cat,varX)
            quartile['range'] = [low,high]
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
                returnDict['overEst'] += appHelper.formatLabel(low) + "-" + appHelper.formatLabel(high) + ", "
            } else {
                returnDict['underEst'] += appHelper.formatLabel(low) + "-" + appHelper.formatLabel(high) + ", "
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
            var catc = catlist[i]
            var cat = cats[c]
            var quartile = {}
            quartile['quarter'] = catc
            quartile['group'] = cat
            var filtData = dataList.filter(function (d) {
                return d[varX] == catc && d['groupByValue'] == cat
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
                returnDict['overEst'] += catc + ", "
            } else {
                returnDict['underEst'] += catc + ", "
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
    })[0]['quarter']
    returnDict['highestQuartile'] = statData['quartiles'].filter(function (d) {
        return d['averageImpact'] == returnDict['highest']
    })[0]['quarter']
    returnDict['lowestRange'] = [appHelper.formatLabel(d3.min(varRange) + diff * (returnDict['lowestQuartile'] - 1)), appHelper.formatLabel(d3.min(varRange) + diff * (returnDict['lowestQuartile']))]
    returnDict['highestRange'] = [appHelper.formatLabel(d3.min(varRange) + diff * (returnDict['highestQuartile'] - 1)), appHelper.formatLabel(d3.min(varRange) + diff * (returnDict['highestQuartile']))]
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


//function to create the color dictionary by assigning a value to each group by variable
function createColorDict() {
    var categories = []

    for (var ind = 0; ind < AppData.length; ind++) {

        var chartRawData = AppData[ind]['Data'].filter(function (d) {
            return d['groupByVarName'] == groupBy
        })
        if (chartRawData.length > 0) {
            var varArray = chartRawData[0]
            var varX = Object.keys(varArray).filter(function(d){return permKeys.indexOf(d) < 0})[0]
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
    var margin = {
        top: 30,
        right: 15,
        bottom: 60,
        left: 60
    }
    var filterChartRawData = AppData[varInd]['Data'].filter(function (d) {
        return d['groupByVarName'] == groupBy && heatMapFilters.indexOf(d['groupByValue']) < 0
    })
    if (filterChartRawData.length == 0) {
        return null;
    }
    var varX = Object.keys(AppData[varInd]['Data'][0]).filter(function(d){return permKeys.indexOf(d) < 0})[0]
    var width = appHelper.getWidth() * 0.74 - margin.left - margin.right;
    var height = 300 - margin.top - margin.bottom;
    var chartLevel = mainApp.append("div").attr("class", "chart-lev").attr("id", "topchlvl-" + varX.replace(/[^a-zA-Z]/g, "")).style("height", "430px")
    var title = chartLevel.append('div').attr('class', 'Title')
    var main = chartLevel.append("div").attr("class", "app").attr("id", "chlvl-" + varX.replace(/[^a-zA-Z]/g, ""))
    var legendGroup = main.append("div").attr("class", "legend").attr("width", width - 100).attr("height", height - 100)

    var filterContainer = main.append('div').attr('class', 'Filtercontainer')
    var total = main.append('div').attr('class', 'Total')
    var chartContainer = main.append('div').attr('class', 'Chartcont')
    var narrative = chartLevel.append('div').attr('class', 'desc').style("height", "560px")
    var narrativeText = narrative.append("div").attr('class', 'desc-text').html("Few Insights will come here")
    var source = main.append('div').attr('class', 'Source')
    var attribution = main.append('div').attr('class', 'Attribution')
    var changePar = ""
    var filterCategories = {}
    filterCategories[varX] = []
    //filtering the data for that particular variable
    filterChartRawData.forEach(function (d) {
        if (filterCategories[varX].indexOf(d['groupByValue']) < 0) {
            filterCategories[varX].push(d['groupByValue'])
        }
    })
    
   //creating the nested data in the format required for d3.stack() function
    var chartRawData = []
    var nestedData = d3.nest().key(function (d) {
        return d[varX]
    }).entries(filterChartRawData)
    nestedData.forEach(function (d) {
        var data = {}
        data[varX] = AppData[varInd]['Type'] == "Continuous" ? parseFloat(d.key) : d.key
        filterCategories[varX].forEach(function (p) {
            var reDatum = d.values.filter(function (k) {
                return k['groupByValue'] == p
            })
            if (reDatum.length == 1) {
                data[p] = reDatum[0]['predictedYSmooth']
            } else {
                data[p] = null
            }
        })
        chartRawData.push(data)
    })
    
    //checking for categorical/continous variables
    var varArray = chartRawData[0]
    if (AppData[varInd]['Type'] == "Continuous") {
        changePar = appHelper.formatLabel(AppData[varInd]['Change'].indexOf("Default") >= 0 ? (AppData[varInd]['Change'].split(":")[1]) : AppData[varInd]['Change'])
        initializeAreaChart()
    } else if (AppData[varInd]['Type'] == "Categorical") {
        changePar = AppData[varInd]['Change'].indexOf("Default") >= 0 ? (AppData[varInd]['Change'].split(":")[1]) : AppData[varInd]['Change']
        initializeBarChart()
    }
    //function to initialize the area chart
    function initializeAreaChart() {
        //creating the svg for the area chart
        svg = appHelper.createChart(chartContainer, width);
        appHelper.setTitle(title, "Impact of Increasing " + varX + " by " + changePar + (AppData[varInd]['Change'].indexOf("Default") >= 0 ? " (1 standard deviation) " : "") + " on " + yVar);
        svg.selectAll('*').remove();
        svg.attr('width', width);
        
        //getting the list of the type variables present in the filtered data and sorting them
        var categories = []
        filterChartRawData.forEach(function (d) {
            if (categories.indexOf(d['groupByValue']) < 0) {
                categories.push(d['groupByValue'])
            }
        })
        categories = categories.sort(appHelper.sortAlphaNum)
        //intializing few local variables
        var widthOffset = 140
        var legendPerRow = parseInt(width / widthOffset)
        var numOfRows = Math.ceil(categories.length / legendPerRow)
        
        //adding the legend for the percentile lines
        var percLegend = svg.append("g")
                            .attr("transform","translate("+(100)+","+(height+numOfRows * 30+55)+")")
        var percLine = percLegend.append("line")
                            .attr("x1", 0)
                            .attr("y1", 0)
                            .attr("x2", 20)
                            .attr("y2",  0)
                            .attr("stroke", "#75787B")
                            .attr("stroke-width","3px")
       var percLine     =  percLegend.append("text")
                                    .text("Percentile")
                                    .attr("x",30)
                                    .attr("y",5)
       
       
       //setting up the height based on the legend height
        svg.attr('height', 300 + numOfRows * 30);
       
       //intializing the scale for the axes 
        var x = d3.scaleLinear().range([margin.left, width])
        var y = d3.scaleLinear().range([height - 30, 0])
        
        //Adding the area functions
        var area = d3.area()
            .x(function (d) {
                return x(d.data[varX])
            })
            .y0(function (d) {
                return y(d[0])
            })
            .y1(function (d) {
                return y(d[1])
            })
            .curve(d3.curveBasis)
        var areaNull = d3.area()
            .x(function (d) {
                return x(d.data[varX])
            })
            .y0(function (d) {
                return y(0)
            })
            .y1(function (d) {
                return y(0)
            })
            .curve(d3.curveBasis)
        
        //creating elements for percentiles and percentile text
        var refLines = svg.append("g").attr("class","percRefLines").attr("transform","translate(10,0)")
        var refLinesText = svg.append("g").attr("class","textRefLines").attr("transform","translate(10,0)")
        
        //adding x axis element
        var xAxisElement = svg.append("g").attr("class", "xAxis").attr("transform", "translate(10," + (height + numOfRows * 30 + 20) + ")")
        
        //adding yaxis element
        var yAxisElement = svg.append("g").attr("class", "yAxis").attr("transform", "translate(60,70)")
        
        //adding group element for area paths
        var pathGroup = svg.append("g")
        
        //adding and formatting x labels
        var xLabel = svg.append("text").attr("class", "label").attr("transform", "translate(" + width / 2 + "," + (height + numOfRows * 30 + 60) + ")").style("text-anchor", "middle").text(varX)
        
        //adding an dformatting y-label
        var yLabel = svg.append("text").attr("class", "label").attr("transform", "translate(15," + (height - 50) + ")rotate(-90)").style("text-anchor", "middle").text("Impact on " + yVar)
        
        //adding element for on hover reference line
        var refLine = svg.append("line")
            .style("opacity", 0)
            .attr("x1", 0)
            .attr("y1", numOfRows * 30)
            .attr("x2", 0)
            .attr("y2", height + numOfRows * 30 + 20)
            .attr("stroke", "#75787B")
        
        //setting up a dummy inital domain for y-axis
        y.domain([-1, 1])
        
        //positioning and calling the y-axix
        var yAxis = d3.axisLeft().scale(y).tickSizeInner(-width + margin.left)
        yAxisElement.call(yAxis)
        
        //function to update the elements, this will help in updating in existing dom elements without creating new
        function drawChart(onlyResizeFlag) {
            //variables to get the range of the y-axis
            var minVal = 0
            var maxVal = 0
            var chartData = []
            
            //extarcting the y-values in to new variable
             chartRawData.filter(function (d) {
                var data = {}
                data[varX] = d[varX]
                filterCategories[varX].forEach(function (c) {
                    data[c] = d[c]
                })
                chartData.push(data)
            })
            
            //creating the quartile data
            var indata = createQuartileData(filterChartRawData, varX, filterCategories[varX])
            //updating the narrative text element based on the quartile data
            narrativeText.html("<ul style='list-style-type:disc' > <li>Across all categories, the impact of <b>" + varX + " </b> on <b>" + yVar + "</b> is lowest from <b> [" + indata['lowestRange'] + "] </b> " + indata['lowestGroup'] + " and highest from <b> [" + indata['highestRange'] + "] </b> " + indata['highestGroup'] + "</li> <li> The impact of <b>" + varX + " </b>  on <b>" + yVar + "</b> from <b> [" + indata['highestRange'] + "]  </b>" + indata['highestGroup'] + " is  " + appHelper.formatLabel(indata['diffMaxOverall']) + " higher than the overall median imapct and " + appHelper.formatLabel(indata['diffMaxMin']) + " higher than the average impact from <b> [" + indata['lowestRange'] + "] </b>" + indata['lowestGroup'] + "</li> </ul>")
            var percDataList = []
            //extracting the percentile data for each group by value
            for (var p in percList){
            var sum = 0
            var sumProp = 0
            for (var c in filterCategories[varX]){
                var cat = filterCategories[varX][c]
                
                var perc =  filterCategories[varX].length== categories.length?getXthPercentaileGlobal(chartRawData, percList[p], cat, varX):getXthPercentaile(chartRawData, percList[p], cat, varX)
                var prop = metaData['proportion'].filter(function(d){
                    return d.name==cat
                })[0]['perc']/100
                sum += perc*prop
            sumProp += prop
            }
            percDataList.push({"percentile":percList[p],"value":appHelper.formatLabel(sum/sumProp)})}
            
            //sorting the data from maximum to minimum
            chartData = chartData.sort(function (a, b) {
                return a[varX] - b[varX]
            })
            
            //setting up the domain for x axis
            x.domain(d3.extent(chartData, function (d) {
                return d[varX]
            }))
            
            //creating stacked values for the area charts
            var stack = d3.stack()
                .keys(filterCategories[varX].sort(appHelper.sortAlphaNum))
                .order(d3.stackOrderAscending)

            
            stack(chartData).forEach(function (d) {
                var sum = 0
                d.forEach(function (c) {
                    sum += c[1]
                    if (d3.min([c[0], c[1]]) < minVal) {

                        minVal = d3.min([c[0], c[1]])
                    }
                    if (d3.max([c[0], c[1]]) > maxVal) {
                        maxVal = d3.max([c[0], c[1]])
                    }
                })
            })

            //setting up the domain for y-axis based on the stacked values
            y.domain([minVal, maxVal])
            var xAxis = d3.axisBottom().scale(x)
            yAxisElement.transition().duration(1000).call(yAxis)
            
            //updating the percentile line positions based on the data check box option
            if (percFlag & filterCategories[varX].length != 0) {
                var refs = refLines.selectAll("line")
                    .data(percDataList).style("opacity",1)
                refs.exit().remove()
                refs.enter().append("line")
                    .style("opacity", 1)
                    .attr("x1", function (d) {
                        return x(d['value'])
                    })
                    .attr("y1", numOfRows * 30+40)
                    .attr("x2", function (d) {
                        return x(d['value'])
                    })
                    .attr("y2", height + numOfRows * 30 + 20)
                    .attr("stroke", "#75787B")
                    .attr("stroke-width", "2px")

                refs.transition().attr("x1", function (d) {
                        return x(d['value'])
                    })
                    .attr("y1", numOfRows * 30+40)
                    .attr("x2", function (d) {
                        return x(d['value'])
                    })
                    .attr("y2", height + numOfRows * 30 + 20)
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
                        return "translate(" + (x(d['value']) - 5) + ",65)"
                    })

                refsText.transition()
                    .text(function (d) {
                        return d['percentile']
                    })
                    .attr('transform', function (d) {
                        return "translate(" + (x(d['value']) - 5) + ",65)"
                    })

            }
            if(filterCategories[varX].length == 0){
                refLines.selectAll("line").style("opacity",0)
                refLinesText.selectAll("text").style("opacity",0)
            }
            
            //removing all the paths before updating
            pathGroup.selectAll(".path").remove()
            //adding the path lines
            var paths = pathGroup.selectAll(".path")
                .data(stack(chartData))
            paths.exit().transition().duration(500).remove()
            paths.enter().append("path")
                .attr("class", function (d, i) {
                    return "path layer" + d.key + " path" + d.key + varInd + " layer" + varInd
                })
                .attr("transform", "translate(10," + (numOfRows * 30 + 40) + ")")
                .on("mouseenter", mouseenter)
                .on("mousemove", mousemove)
                .on("mouseout", mouseout)
                .attr("d", areaNull)
                .transition().duration(1000)
                .attr("d", area)
                .attr("fill", function (d, i) {
                    return colorDict[varX][d.key];
                })
                .attr("stroke","black")
                .attr("stroke-width", "0.5px" )
                .attr("cursor", "pointer")
            paths.attr("class", function (d, i) {
                    return "layer" + d.key + " path" + d.key + varInd + " layer" + varInd
                })
                .on("mouseenter", mouseenter)
                .on("mousemove", mousemove)
                .on("mouseout", mouseout)
                .transition().duration(1000)
                .attr("d", area)
                .attr("fill", function (d, i) {
                    return colorDict[varX][d.key];
                })
                .attr("stroke","black")
                .attr("stroke-width", "0.5px" )
                .attr("transform", "translate(10," + (numOfRows * 30 + 40) + ")")
                .attr("cursor", "pointer");
            
            //adding the grid lines for y-axis
            yAxisElement.selectAll("line").attr("opacity", 0.05).style("stroke-dasharray", ("5, 5"))
            xAxisElement.call(xAxis)
            
            //adding legend buttons in the top
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
            
            //event listeners for the legend buttons
            function legendMouseOver(d, i) {
                d3.selectAll(".layer" + varInd).style("opacity", 0.25)
                d3.selectAll(".path" + d + varInd).style("opacity", 1).style('stroke', 'black').attr("stroke-width", "1.5px")
            }

            function legendMouseOut(d, i) {
                d3.selectAll(".layer" + varInd).style("opacity", 1).attr("stroke-width", "0.5px")
            }
            
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
            //functions to show mosue over values for the area charts
            function mousemove(d) {
                var closestXValue = getClosestxValue()
                tooltip.style("left", function () {
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
                }).style("top", (d3.event.pageY - 28) + "px").style('opacity', 1);
                refLine.transition().style("opacity", 1).attr("x1", x(closestXValue) + 10).attr("x2", x(closestXValue) + 10)
                var filt = chartRawData.filter(function (d) {
                    return d[varX] == closestXValue
                })
                tooltip.html("Within the <span>" + d.key + " </span> group, the model believes that increasing <span>" + varX + "</span> from <span> " + appHelper.formatLabel(filt[0][varX]) + " </span> to <span>" + (appHelper.formatLabel(parseFloat(filt[0][varX]) + parseFloat(changePar))) + "</span> leads to, on average, a <span>" + Math.abs(appHelper.formatLabel(filt[0][d.key])) + (appHelper.formatLabel(filt[0][d.key]) >= 0 ? " increase" : " decrease") + "</span> in " + yVar)

            }

            function mouseout(d) {
                d3.selectAll(".layer" + varInd).style("opacity", 1).attr("stroke-width", "0.5px")
                tooltip.transition().style("opacity", 0);
                showingTooltip = false;
                refLine.transition().style("opacity", 0)
            }

            function mouseenter(d, i) {

                var closestXValue = getClosestxValue()
                setTimeout(function () {
                    showingTooltip = true;
                }, 300)
                var filt = chartRawData.filter(function (d) {
                    return d[varX] == closestXValue
                })
                tooltip.attr('class', "colorClass" + varInd + i);
                tooltip.transition().duration(200).delay(100).style("opacity", 1);
                tooltip.html("Within the <span>" + d.key + " </span> group, the model believes that increasing <span>" + varX + "</span> from <span> " + appHelper.formatLabel(filt[0][varX]) + " </span> to <span>" + (appHelper.formatLabel(parseFloat(filt[0][varX]) + parseFloat(changePar))) + "</span> leads to, on average, a <span>" + Math.abs(appHelper.formatLabel(filt[0][d.key])) + (appHelper.formatLabel(filt[0][d.key]) >= 0 ? " increase" : " decrease") + "</span> in " + yVar)
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
                    .style("top", (d3.event.pageY - 28) + "px")
                tooltip.style("color", colorDict[varX][d.key])

                d3.selectAll(".layer" + varInd).style("opacity", "0.4")
                d3.select(this).style("opacity", 1).style('stroke', 'black').attr("stroke-width", "0.5px")
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
        
        //initializing local varaibles required for the bar plot
        var barData = varArray
        var maxHeight = 0
        
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
        //extarcting the group by varaibles
        var categoryData = getCategories(chartRawData)
        // creating the svg element for bar charts
        svg = appHelper.createChart(chartContainer, width);
        //intializing and upadting the list of variables
        var categories = []
        filterChartRawData.forEach(function (d) {
            if (categories.indexOf(d['groupByValue']) < 0) {
                categories.push(d['groupByValue'])
            }
        })
        //settign up title for the chart
        appHelper.setTitle(title, "Sensitivity Plot: Impact of changing " + varX + " to " + changePar);
        
        //sorting the group by variables
        categories = categories.sort(appHelper.sortAlphaNum)
        //setting up the layout for the chart
        svg.selectAll('*').remove();
        svg.attr('width', width)
        var widthOffset = 140
        var legendPerRow = parseInt(width / widthOffset)
        var numOfRows = Math.ceil(categories.length / legendPerRow)
        svg.attr('height', 300 + numOfRows * 30);
        
        
        //intializing the axes
        var x = d3.scaleBand().range([margin.left, width]).paddingInner(0.25)
        var x1 = d3.scaleBand().padding(0.25)
        var y = d3.scaleLinear().range([height, 0])
        
        //creating the axes elements
        svg.append("g").attr("class", "xAxis").attr("transform", "translate(10," + (height + numOfRows * 30 + 20) + ")")
        var yAxisElement = svg.append("g").attr("class", "yAxis").attr("transform", "translate(60,70)")
        
        //creating the group element for the bars
        var barGroup = svg.append("g").attr("transform", "translate(0,70)")
        
        //creating elements for axes labels
        var xLabel = svg.append("text").attr("class", "label").attr("transform", "translate(" + width / 2 + "," + (numOfRows * 30) + ")").style("text-anchor", "middle").text(varX)
        var yLabel = svg.append("text").attr("class", "label").attr("transform", "translate(15," + (height - 50) + ")rotate(-90)").style("text-anchor", "middle").text("Impact on " + yVar)
        
        
        var refLine = svg.append("line")
            .style("opacity", 0)
            .attr("x1", 0)
            .attr("y1", numOfRows * 30)
            .attr("x2", 0)
            .attr("y2", height + numOfRows * 30 + 10)
            .attr("stroke", "#75787B")
        
        //setting up dummy domain value for y-axis and calling the y-axiss
        y.domain([-1, 1])
        var yAxis = d3.axisLeft().scale(y).tickSizeInner(-width + margin.left)
        yAxisElement.call(yAxis)
         
        //function to update the elements, this will help in updating in existing dom elements without creating new
        function drawChart() {
            
            
            svg = d3.select("#chlvl-" + varX.replace(/[^a-zA-Z]/g, "")).select("svg")
            
            //filtering the categories available for this variable
            var newCategories = categories.filter(function (k) {
                return filterCategories[varX].indexOf(k) >= 0
            })
            
            //intializaing the variables to calculate the range of y-values and updating the y-domain
            var minVal = 0
            var maxVal = 0
            chartRawData.forEach(function (c) {
                var eachMin = d3.min(newCategories, function (d) {
                    return c[d]
                })
                var eachMax = d3.max(newCategories, function (d) {
                    return c[d]
                })
                if (eachMin < minVal) {
                    minVal = eachMin
                }
                if (eachMax > maxVal) {
                    maxVal = eachMax
                }
            })

            minVal = minVal > 0 ? minVal * 0.9 : minVal * 1.1
            maxVal = maxVal > 0 ? maxVal * 1.1 : maxVal * 0.9
            var chartRange = [minVal > 0 ? 0 : minVal, maxVal < 0 ? 0 - (minVal * 0.1) : maxVal]
            y.domain(chartRange)
            //setting up x-domain
            x.domain(categoryData)
            x1.domain(newCategories).rangeRound([0, x.bandwidth()])
            
            //creatin quartile data  
            var indata = createQuartileDataCategory(filterChartRawData, varX, filterCategories[varX], categoryData)
            //updating the narrative text based on the quartile values
            narrativeText.html("<ul style='list-style-type:disc' > <li>Across all categories, the impact of <b>" + varX + " </b> on <b>" + yVar + "</b> is lowest for <b> " + indata['lowestQuartile'] + " </b>  " + indata['lowestGroup'] + " and highest for <b> " + indata['highestQuartile'] + " </b> " + indata['highestGroup'] + "</li> <li> The impact of <b>" + varX + " </b>  on <b>" + yVar + "</b> for <b>" + indata['highestQuartile'] + " </b> " + indata['highestGroup'] + " is  " + appHelper.formatLabel(indata['diffMaxOverall']) + " higher than the overall median impact and " + appHelper.formatLabel(indata['diffMaxMin']) + " higher than the average impact from <b>" + indata['lowestQuartile'] + "</b> " + indata['lowestGroup'] + "</li>  </ul>")

            //calling and updating the axes elements
            var xAxis = d3.axisBottom().scale(x)
            yAxisElement.transition().duration(1000).call(yAxis)
            
            //removing and updating the bar elements on filtering
            barGroup.selectAll("g").remove()
            barGroup.selectAll("g")
                .data(categoryData)
                .enter().append("g")
                .attr("transform", function (d) {
                    return "translate(" + x(d) + ",0)"
                })
                .selectAll("rect")
                .data(function (d) {

                    return newCategories.map(function (p) {
                        if (filterCategories[varX].indexOf(p) >= 0) {
                            var dict = {}
                            dict["parent"] = d
                            dict["key"] = p
                            dict["value"] = chartRawData.filter(
                                function (k) {
                                    return k[varX] == d
                                }
                            )[0][p]
                            return dict
                        }
                    })
                })
                .enter().append("rect")
                .attr("class", function (d) {
                    return "rect" + d.key.replace(/[^a-zA-Z]/g, "")
                })
                .attr("x", function (d) {
                    return x1(d["key"])
                })
                .attr("y", function (d) {
                    return y(d["value"]) >= y(0) ? y(0) : y(d["value"])
                })
                .attr("width", x1.bandwidth())
                .attr("height", function (d) {

                    return y(d["value"]) - y(0) >= 0 ? y(d["value"]) - y(0) : y(0) - y(d["value"])
                })
                .attr("fill", function (d, i) {

                    return colorDict[varX][d['key']]
                })
                .on("mouseover", mouseover)
                .on("mousemove", mousemove)
                .on("mouseout", mouseout)
                .style("cursor", "pointer")
                .style("stroke","#000")
                .style("stroke-width","1px")

            //adding grid lines and positioning axes
            yAxisElement.selectAll("line").attr("opacity", 0.15).style("stroke-dasharray", ("4, 4"))

            svg.select(".xAxis").call(xAxis).attr("transform", "translate(0," + (y(0) + 70) + ")")
            svg.select(".xAxis")
                .selectAll("text")
                .attr("x", function (d) {
                    return 0;
                })
                .attr("transform", "translate(0," + (y(chartRange[0]) - y(0)) + ")")
                .call(wrap, x.bandwidth()*0.9)
            
            
            //function to display tooltip on hover of bar elements
            function mouseover(d, i) {
                tooltip.attr('class', "colorClass" + varInd + i);
                tooltip.transition().duration(200).delay(100).style("opacity", 1);

                tooltip.html("Within the <span>" + d.key + " </span> group, the model believes that increasing <span>" + varX + "</span> from <span> " + d['parent'] + " </span> to <span>" + changePar + "</span> leads to, on average, a <span>" + Math.abs(appHelper.formatLabel(d.value)) + (appHelper.formatLabel(d.value) >= 0 ? " increase" : " decrease") + "</span> in " + yVar)
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
                    .style("top", (d3.event.pageY - 28) + "px")
                tooltip.style("color", colorDict[varX][d])

                d3.selectAll(".layer" + varInd).style("opacity", "0.4")
                d3.select(this).style("opacity", 1).style('stroke', 'black').attr("stroke-width", "1px")
            }

            function mousemove(d, i) {
                tooltip.attr('class', "colorClass" + varInd + i);
                tooltip.transition().duration(200).delay(100).style("opacity", 1);
                tooltip.html("Within the <span>" + d.key + " </span> group, the model believes that increasing <span>" + varX + "</span> from <span> " + d['parent'] + " </span> to <span>" + changePar + "</span> leads to, on average, a <span>" + Math.abs(appHelper.formatLabel(d.value)) + (appHelper.formatLabel(d.value) >= 0 ? " increase" : " decrease") + "</span> in " + yVar)
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
                    .style("top", (d3.event.pageY - 28) + "px")
                tooltip.style("color", colorDict[varX][d.key])

                d3.selectAll(".layer" + varInd).style("opacity", "0.4")
                d3.select(this).style("opacity", 1).style('stroke', 'black').attr("stroke-width", "1px")
            }

            function mouseout(d, i) {
                tooltip.transition().style("opacity", 0);
                showingTooltip = false;
                refLine.transition().style("opacity", 0)

                d3.select(this).attr("stroke-width", "0px")
            }
            
            //creating legend buttons in the top
            var legend = legendGroup.selectAll("button")
                .data(categories)
                .enter().append("button")
                .attr("class", "legend-btn")
                .style("background-color", function (d) {
                    return colorDict[varX][d]
                })
                .on("click", legendClick)
                .on("mouseover", legendMouseOver)
                .on("mouseout", legendMouseOut)
                .style("cursor", "pointer")
                .html(function (d) {
                    return d
                })
            
            
            //event listeners for the legend buttons
            function legendMouseOver(d, i) {
                d3.selectAll(".rect" + d.replace(/[^a-zA-Z]/g, "")).style('stroke', 'black').attr("stroke-width", "1px")
            }

            function legendMouseOut(d, i) {
                d3.selectAll(".rect" + d.replace(/[^a-zA-Z]/g, "")).style('stroke', 'black').attr("stroke-width", "0px")
            }

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
            
            //function to wrap the x-label text to the width
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
                    maxHeight = d3.max([430+lineNumber*20,maxHeight])
                    
                    d3.select("#topchlvl-"+varX.replace(/[^a-zA-Z]/g, "")).style("height",maxHeight+"px")
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
    /*for (var ind = 0; ind < AppData.length; ind++) {
        loopVariables(ind)
    }*/

}
