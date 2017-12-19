//****************************************************************************
//Word Cloud
//Source: https://bl.ocks.org/blockspring/847a40e23f68d6d7e8b5
//Purpose: to understand the usage of Word Cloud and how to call it using D3.
//****************************************************************************
var checkif_svg_exists = document.getElementById('bagg');
if (checkif_svg_exists != undefined || checkif_svg_exists != null)
{
    d3.selectAll("#bagg").remove();
}

var margin = {top: 20, right: 20, bottom: 40, left: 20},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

//read the data from csv
d3.csv("/static/data.csv", function(error, data) {

    var categories = d3.keys(d3.nest().key(function(d) { return d.category; }).map(data));
    var color = d3.scale.ordinal().range(["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854"]);
    var fontSize = d3.scale.pow().exponent(5).domain([0,1]).range([10,80]);

    var layout = d3.layout.cloud()
        .timeInterval(10)
        .size([width, height])
        .words(data)
        .rotate(function(d) { return 0; })
        .font('monospace')
        .fontSize(function(d,i) { return fontSize(Math.random()); })
        .text(function(d) { return d.password; })
        .spiral("archimedean")
        .on("end", draw)
        .start();

    //create an empty svg
    var svg = d3.select('#bagword').append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .attr("id","bagg")
        .attr("transform","translate(-10,-250)")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var wordcloud = svg.append("g")
        .attr('class','wordcloud')
        .attr("transform", "translate(" + width/2 + "," + height/2 + ")");

    var x0 = d3.scale.ordinal()
        .rangeRoundBands([0, width], .1)
        .domain(categories);

    var xAxis = d3.svg.axis()
        .scale(x0)
        .orient("bottom");

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
        .selectAll('text')
        .style('font-size','20px')
        .style('fill',function(d) { return color(d); })
        .style('font','sans-serif');

    function draw(words) {
        wordcloud.selectAll("text")
            .data(words)
            .enter().append("text")
            .attr('class','word')
            .style("font-size", function(d) { return d.size + "px"; })
            .style("font-family", function(d) { return d.font; })
            .style("fill", function(d) {
                var paringObject = data.filter(function(obj) { return obj.password === d.text});
                return color(paringObject[0].category);
            })
            .attr("text-anchor", "middle")
            .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
            .text(function(d) { return d.text; });
    };

});
