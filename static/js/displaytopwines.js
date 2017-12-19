//this function calls the top wines in the dashboard containing barchart and piechart format.
function displaytopwines()
{
    var elem = document.getElementById('winesList');
    var elem1 = document.getElementById('text');
    if (elem !=null)
    {
        elem.parentNode.removeChild(elem);
    }

    if(elem1 != null){
        elem1.parentNode.removeChild(elem1);
    }

    var elem = document.getElementById('SVG_collapsibletree');
    var barChartTemp = document.getElementById('SVG_barchart');
    if (elem !=null || barChartTemp != null)
    {
        elem.parentNode.removeChild(elem);
        barChartTemp.parentNode.removeChild(barChartTemp);
    }

    var ddlView = document.getElementById("ddlViewBy");
    if (ddlView.value!="select"){

        var jsonStr= "http://127.0.0.1:8004/getTopWines/" + ddlView.value;
        // load the data
        d3.json(jsonStr, function(error, data) {
            //div reference
            var div_Visusalization = document.getElementById("wineDropDown");

            var span     = document.createElement("span");
            span.id="text";
            var oText = document.createTextNode("Top " + ddlView.value + " Elements: ");
            span.appendChild(oText);
            div_Visusalization.appendChild(span);

            // create elements <table> and a <tbody>
            var select     = document.createElement("select");
            select.id = "winesList";

            div_Visusalization.appendChild(select);


            //add columns
            for (var j = 0; j < data.winenames.length; j++) {

                if(j==0){
                    var option = document.createElement("option");
                    option.value = "select";
                    option.text = "select";
                    select.appendChild(option);
                }

                var option = document.createElement("option");
                option.value = data.winenames[j];
                option.text = data.winenames[j];
                select.appendChild(option);

            }

            select.setAttribute("onChange", "callBarChart(this)");
        });
    }
}

//this function call the dashboard.
function callBarChart(obj)
{

    var selectedValue = document.getElementById("winesList");
    var width = 1200;
    var height = 400;
    if(selectedValue.options[selectedValue.selectedIndex].value != null && selectedValue.options[selectedValue.selectedIndex].value != "select"){
        var elem = document.getElementById('SVG_collapsibletree');
        var dash = document.getElementById('histoGramDashboard');
        var pie = document.getElementById('pieChartDashboard');
        var table = document.getElementById('tableDashboard');
        if (elem !=null)
        {
            elem.parentNode.removeChild(elem);
        }
        if(dash != null){
            dash.parentNode.removeChild(dash);
            pie.parentNode.removeChild(pie);
            table.parentNode.removeChild(table);
        }
        var wineName = selectedValue.options[selectedValue.selectedIndex].value;

        var freqData= "http://127.0.0.1:8004/getWineCountries/" + wineName;

        d3.json(freqData, function(error, data) {
            dashboard('#dashboard',data, wineName);
        });
    }
}
