//****************************************************************************
// Treemap
//Source: https://bl.ocks.org/mbostock/4063582
//Purpose: to understand the usage of tree map and how to call it
//****************************************************************************
function displayTreeMap(Country, wineName){

    var elem = document.getElementById('dvTreeMap');
        if (elem !=null)
        {
            elem.parentNode.removeChild(elem);
        }

    //add a note to read the tree map
    var elemNote = document.getElementById('note');
    if (elemNote ==null)
    {
        var note = document.createElement('p');// make a hr
        note.id = "note";
        note.align = "center";
        note.innerHTML = "<b> Note: </b> Different colors refers different provinces. Size of a rectangle denotes price, <br/> the text in the rectangle is the winery region name. (Hover to get more details)";
        document.getElementById('Treemap').appendChild(note) // append a paragraph contain note
    }
    var margin = {top: 20, right: 120, bottom: 20, left: 120},
            width = 960 - margin.right - margin.left,
            height = 800 - margin.top - margin.bottom;

    var svg = d3version4.select("#Treemap").append("svg")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
            .attr("id","dvTreeMap")
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var fader = function(color) { return d3version4.interpolateRgb(color, "#fff")(0.2); },
        color = d3version4.scaleOrdinal(d3version4.schemeCategory20.map(fader)),
        format = d3version4.format(",d");

    var treemap = d3version4.treemap()
        .tile(d3version4.treemapResquarify)
        .size([width, height])
        .round(true)
        .paddingInner(1);

    // create final link to tree map
    var jsonStr= "http://127.0.0.1:8004/getTreeMap/" + Country +"/test/"+ wineName

    //read json
    d3version4.json(jsonStr, function(error, data) {
        if (error) throw error;
        var root = d3version4.hierarchy(data)
            .eachBefore(function(d) { d.data.id = (d.parent ? d.parent.data.id + "." : "") + d.data.name; })
            .sum(sumBySize)
            .sort(function(a, b) { return b.height - a.height || b.value - a.value; });

        treemap(root);

        var cell = svg.selectAll("g")
            .data(root.leaves())
            .enter().append("g")
            .attr("transform", function(d) { return "translate(" + d.x0 + "," + d.y0 + ")"; });

        cell.append("rect")
            .attr("id", function(d) { return d.data.id; })
            .attr("width", function(d) { return d.x1 - d.x0; })
            .attr("height", function(d) { return d.y1 - d.y0; })
            .attr("fill", function(d) { return color(d.parent.data.id); });

        cell.append("clipPath")
            .attr("id", function(d) { return "clip-" + d.data.id; })
            .append("use")
            .attr("xlink:href", function(d) { return "#" + d.data.id; });

        cell.append("text")
            .attr("clip-path", function(d) { return "url(#clip-" + d.data.id + ")"; })
            .selectAll("tspan")
            .data(function(d) { return d.data.name.split(/(?=[A-Z][^A-Z])/g); })
            .enter().append("tspan")
            .attr("x", 4)
            .attr("y", function(d, i) { return 13 + i * 10; })
            .text(function(d) { return d; });

        cell.append("title")
            .text(function(d) { return d.data.id + "\n" + format(d.value); });

        d3version4.selectAll("input")
            .data([sumBySize, sumByCount], function(d) { return d ? d.name : this.value; })
            .on("change", changed);

        var timeout = d3version4.timeout(function() {
            d3version4.select("input[value=\"sumByCount\"]")
                .property("checked", true)
                .dispatch("change");
        }, 2000);

        function changed(sum) {
            timeout.stop();

            treemap(root.sum(sum));

            cell.transition()
                .duration(750)
                .attr("transform", function(d) { return "translate(" + d.x0 + "," + d.y0 + ")"; })
                .select("rect")
                .attr("width", function(d) { return d.x1 - d.x0; })
                .attr("height", function(d) { return d.y1 - d.y0; });
        }
    });

    function sumByCount(d) {
        return d.children ? 0 : 1;
    }

    function sumBySize(d) {
        return d.size;
    }
}
