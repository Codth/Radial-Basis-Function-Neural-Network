function get_params(Data) {
	var python = require("python-shell")
	var path = require("path")
	var fs = require('fs');
	var remote = require('electron').remote;

	//console.log(Data);

	//make sure dataset exists
	if (Data == "" || Data == undefined) {
		document.getElementById("textArea").value = "No dataset selected";
		return;
	}

	var win = remote.getCurrentWindow();
	win.webContents.session.clearCache(function(){
		//some callback.
	});

	//deleteing old figure
	//fs.unlinkSync('figure.png');

	var NumOfCenters1 = document.getElementById("centers1").checked
	var NumOfCenters2 = document.getElementById("centers2").checked
	var ManualCenters = document.getElementById("numCen").value

	var CenterLocations1 = document.getElementById("randomDp").checked
	var CenterLocations2 = document.getElementById("kMeans").checked

	var width1 = document.getElementById("sigma").checked
	var width2 = document.getElementById("stdDev").checked

	var options = {
		scriptPath : path.join(__dirname, '/engine/'),
		args : [NumOfCenters1, NumOfCenters2, ManualCenters, CenterLocations1, CenterLocations2, width1, width2, Data]
	}


	var RBFN = new python('RadialBasisNN.py', options);

	var img = document.getElementById("image");
	img.setAttribute("src", "graph2.png");
	document.getElementById("textArea").value = "";
	document.getElementById("textArea").value += 'Calculating...';

	RBFN.on('message', function(message) {
		//swal(message);
		if (document.getElementById("textArea").value == 'Calculating...') {
			document.getElementById("textArea").value = "";
		}
		document.getElementById("textArea").value += message;
		if (NumOfCenters2 == true) {
			img = document.getElementById("image");
			img.setAttribute("src", "graph2.png");
		}
		else {
			img = document.getElementById("image");
			img.setAttribute("src", "figure.png?" + new Date().getTime());
		}
	})
}