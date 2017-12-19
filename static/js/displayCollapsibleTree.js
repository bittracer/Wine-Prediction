//****************************************************************************
// Collapsible tree
//Source: https://bl.ocks.org/mbostock/4339083
//Purpose: to understand the usage of collapsible tree and how to call it.
//****************************************************************************
function displayCollapsibleTree(Country, wineName)
                        {
                            if (Country!=null)
                            {

                                var elem = document.getElementById('SVG_collapsibletree');

                                if (elem !=null)
                                {
                                    elem.parentNode.removeChild(elem);
                                }
                                //add a line break and a note to read collapsible tree
                                var elemLineBreak = document.getElementById('linebreak');
                                if (elem ==null)
                                {
                                    var line = document.createElement('hr');// make a hr
                                    line.width = "1150px";
                                    line.id = "linebreak";
                                    document.getElementById('dvcollapsibletree').appendChild(line) // append a line break

                                    var note = document.createElement('p');// make a hr
                                    note.id = "note";
                                    note.align = "center";
                                    note.innerHTML = "<b> Collapsible tree hierarchy: </b> Country -> Province -> Region -> Winery -> Price";
                                    document.getElementById('dvcollapsibletree').appendChild(note) // append a paragraph contain note

                                }
                                var margin = {top: 20, right: 120, bottom: 20, left: 120},
                                    width = 960 - margin.right - margin.left,
                                    height = 800 - margin.top - margin.bottom;

                                var i = 0,
                                    duration = 750,
                                    root;

                                var tree = d3.layout.tree()
                                    .size([height, width]);

                                var diagonal = d3.svg.diagonal()
                                    .projection(function(d) { return [d.y, d.x]; });

                                //create an empty svg
                                var svg = d3.select("#dvcollapsibletree").append("svg")
                                    .attr("width", width + margin.right + margin.left)
                                    .attr("height", height + margin.top + margin.bottom)
                                    .attr("id","SVG_collapsibletree")
                                    .append("g")
                                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

                                //display loading timer (spinner)
                                //Available: http://bl.ocks.org/eesur/cf81a5ea738f85732707
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

                                // loader settings
                                var target = document.getElementById('dvcollapsibletree');
                                // trigger loader
                                var spinner = new Spinner(opts).spin(target);

                                var jsonStr= "http://127.0.0.1:8004/getCollapsibleTree/" + Country +"/test/"+ wineName
                                d3.json(jsonStr, function(error, flare) {
                                    if (error) throw error;

                                    // stop the loader
                                    spinner.stop();

                                    root = flare;
                                    root.x0 = height / 2;
                                    root.y0 = 0;

                                    function collapse(d) {
                                        if (d.children) {
                                            d._children = d.children;
                                            d._children.forEach(collapse);
                                            d.children = null;
                                        }
                                    }

                                    root.children.forEach(collapse);
                                    update(root);
                                });

                                d3.select(self.frameElement).style("height", "800px");

                                function update(source) {

                                    // Compute the new tree layout.
                                    var nodes = tree.nodes(root).reverse(),
                                        links = tree.links(nodes);

                                    // Normalize for fixed-depth.
                                    nodes.forEach(function(d) { d.y = d.depth * 180; });

                                    // Update the nodes…
                                    var node = svg.selectAll("g.node")
                                        .data(nodes, function(d) { return d.id || (d.id = ++i); });

                                    // Enter any new nodes at the parent's previous position.
                                    var nodeEnter = node.enter().append("g")
                                        .attr("class", "node")
                                        .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
                                        .on("click", click);

                                    nodeEnter.append("circle")
                                        .attr("r", 1e-6)
                                        .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

                                    nodeEnter.append("text")
                                        .attr("x", function(d) { return d.children || d._children ? -10 : 10; })
                                        .attr("dy", ".35em")
                                        .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
                                        .text(function(d) { return d.name; })
                                        .style("fill-opacity", 1e-6);

                                    // Transition nodes to their new position.
                                    var nodeUpdate = node.transition()
                                        .duration(duration)
                                        .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

                                    nodeUpdate.select("circle")
                                        .attr("r", 4.5)
                                        .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

                                    nodeUpdate.select("text")
                                        .style("fill-opacity", 1);

                                    // Transition exiting nodes to the parent's new position.
                                    var nodeExit = node.exit().transition()
                                        .duration(duration)
                                        .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
                                        .remove();

                                    nodeExit.select("circle")
                                        .attr("r", 1e-6);

                                    nodeExit.select("text")
                                        .style("fill-opacity", 1e-6);

                                    // Update the links…
                                    var link = svg.selectAll("path.link")
                                        .data(links, function(d) { return d.target.id; });

                                    // Enter any new links at the parent's previous position.
                                    link.enter().insert("path", "g")
                                        .attr("class", "link")
                                        .style("stroke", function(d){
                                            console.log(d.target.type);
                                            if(d.target.type == "min"){
                                                return "#807dba"
                                            }else if(d.target.type =="median"){
                                                return "#e08214"
                                            }else if(d.target.type == "max"){
                                                return "#41ab5d"
                                            }else if(d.target.type == "many"){
                                                return "red"
                                            }
                                        })
                                        .style("stroke-width", function(d){
                                            console.log(d.target.type);
                                            if(d.target.type == "none"){
                                                return "1"
                                            }else {
                                                return "2"
                                            }
                                        })
                                        .attr("d", function(d) {
                                            var o = {x: source.x0, y: source.y0};
                                            return diagonal({source: o, target: o});
                                        });

                                    // Transition links to their new position.
                                    link.transition()
                                        .duration(duration)
                                        .attr("d", diagonal);

                                    // Transition exiting nodes to the parent's new position.
                                    link.exit().transition()
                                        .duration(duration)
                                        .attr("d", function(d) {
                                            var o = {x: source.x, y: source.y};
                                            return diagonal({source: o, target: o});
                                        })
                                        .remove();

                                    // Stash the old positions for transition.
                                    nodes.forEach(function(d) {
                                        d.x0 = d.x;
                                        d.y0 = d.y;
                                    });
                                }

                                // Toggle children on click.
                                function click(d) {
                                    if (d.children) {
                                        d._children = d.children;
                                        d.children = null;
                                    } else {
                                        d.children = d._children;
                                        d._children = null;
                                    }
                                    update(d);
                                }
                            }

                        }