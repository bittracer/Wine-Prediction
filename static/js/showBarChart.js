//****************************************************************************
// BarChart
//Source: http://bl.ocks.org/juan-cb/faf62e91e3c70a99a306
//Purpose: to understand the usage of Barchart and how to call it
//****************************************************************************
function showBarChart(){

    var checkif_svg_exists = document.getElementById('SVG_MLbarchart');
    if (checkif_svg_exists != undefined || checkif_svg_exists != null)
    {
        d3.selectAll("#SVG_MLbarchart").remove();
    }
    var div = d3.select("#ShowMLVisualization").append("div").attr("class", "toolTip");

    //the final link to api.
    var jsonStr = "http://127.0.0.1:8004/getMLBarChart/" + words;

    words = "";
    tmp = "";

    var tooltip = d3.select("body").append("div").attr("class", "toolTip");

    //add a spinner (loader) for displaying data loadin
    ////Available: http://bl.ocks.org/eesur/cf81a5ea738f85732707
    var opts = {
        lines: 13, // The number of lines to draw
        length: 28, // The length of each line
        width: 14, // The line thickness
        radius: 42, // The radius of the inner circle
        scale: 1, // Scales overall size of the spinner
        corners: 1, // Corner roundness (0..1)
        color: '#0b0707', // #rgb or #rrggbb or array of colors
        opacity: 0.25, // Opacity of the lines
        rotate: 0, // The rotation offset
        direction: 1, // 1: clockwise, -1: counterclockwise
        speed: 1, // Rounds per second
        trail: 60, // Afterglow percentage
        fps: 20, // Frames per second when using setTimeout() as a fallback in IE 9
        zIndex: 2e9, // The z-index (defaults to 2000000000)
        className: 'spinner', // The CSS class to assign to the spinner
        top: '50%', // Top position relative to parent
        left: '50%', // Left position relative to parent
        shadow: false, // Whether to render a shadow
        position: 'absolute' // Element positioning
    };
    // trigger loader
    var target = document.getElementById('ShowMLVisualization');
    var spinner = new Spinner(opts).spin(target);

    d3.json(jsonStr, function(error, data) {
        if (error) throw error;

        spinner.stop();

        //read the data from the api call
        data.forEach(function(d) {
            d.wine = d.wine;
            d.probability = +d.probability;
        });
        var axisMargin = 20,
            margin = 40,
            valueMargin = 4,
            width = 700,
            height = 500,
            barHeight = (height-axisMargin-margin*2)* 0.4/data.length,
            barPadding = (height-axisMargin-margin*2)*0.6/data.length,
            data, bar, svg, scale, xAxis, labelWidth = 0;

        max = d3.max(data, function(d) { return d.probability; });

        svg = d3.select('#ShowMLVisualization')
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("id","SVG_MLbarchart");

        bar = svg.selectAll("g")
            .data(data)
            .enter()
            .append("g");

        bar.attr("class", "bar")
            .attr("cx",0)
            .attr("transform", function(d, i) {
                return "translate(" + margin + "," + (i * (barHeight + barPadding) + barPadding) + ")";
            });

        bar.append("text")
            .attr("class", "label")
            .attr("y", barHeight / 2)
            .attr("dy", ".35em") //vertical align middle
            .text(function(d){
                return d.wine;
            }).each(function() {
            labelWidth = Math.ceil(Math.max(labelWidth, this.getBBox().width));
        });

        scale = d3.scale.linear()
            .domain([0, max])
            .range([0, width - margin*2 - labelWidth]);

        xAxis = d3.svg.axis()
            .scale(scale)
            .tickSize(-height + 2*margin + axisMargin)
            .orient("bottom");

        bar.append("rect")
            .attr("transform", "translate("+labelWidth+", 0)")
            .attr("height", barHeight)
            .attr("width", function(d){
                return scale(d.probability);
            });

        bar.append("text")
            .attr("class", "value")
            .attr("y", barHeight / 2)
            .attr("dx", -valueMargin + labelWidth) //margin right
            .attr("dy", ".35em") //vertical align middle
            .attr("text-anchor", "end")
            .text(function(d){
                return (d.probability+"%");
            })
            .attr("x", function(d){
                var width = this.getBBox().width;
                return Math.max(width + valueMargin, scale(d.probability));
            });

        bar
            .on("mousemove", function(d){
                div.style("left", d3.event.pageX+10+"px");
                div.style("top", d3.event.pageY-25+"px");
                div.style("display", "inline-block");
                div.html((d.wine)+"<br>"+(d.probability)+"%");
            });
        bar
            .on("mouseout", function(d){
                div.style("display", "none");
            });

        svg.insert("g",":first-child")
            .attr("class", "axisHorizontal")
            .attr("transform", "translate(" + (margin + labelWidth) + ","+ (height - axisMargin - margin)+")")
            .call(xAxis);

        // text label for the y axis
          svg.append("text")
              .attr("transform", "rotate(-90)")
              .attr("y",0)
              .attr("x",0 - (height / 2))
              .attr("dy", "1em")
              .style("text-anchor", "middle")
              .attr("class","lead")
              .text("Name of wines");

        // text label for the x axis
          svg.append("text")
              .attr("transform",
                    "translate(" + (width/2) + " ," +
                                   490 + ")")
              .style("text-anchor", "middle")
              .attr("class","lead")
              .text("Prediction probability");

    });

}