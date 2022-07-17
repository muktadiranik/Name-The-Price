function predict() {
    console.log("");
    var LotArea = document.getElementById("uiLotArea");
    var TotalBsmtSF = document.getElementById("uiTotalBsmtSF");
    var GrLivArea = document.getElementById("uiGrLivArea");
    var FullBath = document.getElementById("uiFullBath");
    var BedroomAbvGr = document.getElementById("uiBedroomAbvGr");
    var KitchenAbvGr = document.getElementById("uiKitchenAbvGr");
    var CentralAir = document.getElementById("uiCentralAir");
    var Garage = document.getElementById("uiGarage");
    var Porch = document.getElementById("uiPorch");
    var Alley = document.getElementById("uiAlley");
    var Street = document.getElementById("uiStreet");
    var Utilities = document.getElementById("uiUtilities");
    var Neighborhood = document.getElementById("uiNeighborhood");
    var MSZoning = document.getElementById("uiMSZoning");
    var BldgType = document.getElementById("uiBldgType");
    var HouseStyle = document.getElementById("uiHouseStyle");
    var RoofStyle = document.getElementById("uiRoofStyle");
    var ExterCond = document.getElementById("uiExterCond");
    var Heating = document.getElementById("uiHeating");
    var Electrical = document.getElementById("uiElectrical");
    var PoolQC = document.getElementById("uiPoolQC");

    var Price = document.getElementById("Price");

    var url = "http://127.0.0.1:5000/predict";

    $.post(url, {
        LotArea: parseFloat(LotArea.value),
        TotalBsmtSF: parseFloat(TotalBsmtSF.value),
        GrLivArea: parseFloat(GrLivArea.value),
        FullBath: parseFloat(FullBath.value),
        BedroomAbvGr: parseFloat(BedroomAbvGr.value),
        KitchenAbvGr: parseFloat(KitchenAbvGr.value),
        Street: Street.value,
        Alley: Alley.value,
        Utilities: Utilities.value,
        CentralAir: CentralAir.value,
        Garage: Garage.value,
        Porch: Porch.value,
        Neighborhood: Neighborhood.value,
        MSZoning: MSZoning.value,
        BldgType: BldgType.value,
        HouseStyle: HouseStyle.value,
        RoofStyle: RoofStyle.value,
        ExterCond: ExterCond.value,
        Heating: Heating.value,
        Electrical: Electrical.value,
        PoolQC: PoolQC.value
    }, function(data, status) {
        console.log(data.price);
        Price.innerHTML = `${data.price.toString()} $`;
        console.log(status);

    });

}

function onPageLoad() {
    console.log("");
    var url = "http://127.0.0.1:5000/Neighborhood";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Neighborhood = data.Neighborhood;
            var uiNeighborhood = document.getElementById("uiNeighborhood");
            $('#uiNeighborhood').empty();
            for (var i in Neighborhood) {
                var opt = new Option(Neighborhood[i]);
                $('#uiNeighborhood').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/MSZoning";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var MSZoning = data.MSZoning;
            var uiMSZoning = document.getElementById("uiMSZoning");
            $('#uiMSZoning').empty();
            for (var i in MSZoning) {
                var opt = new Option(MSZoning[i]);
                $('#uiMSZoning').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/BldgType";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var BldgType = data.BldgType;
            var uiBldgType = document.getElementById("uiBldgType");
            $('#uiBldgType').empty();
            for (var i in BldgType) {
                var opt = new Option(BldgType[i]);
                $('#uiBldgType').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/HouseStyle";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var HouseStyle = data.HouseStyle;
            var uiHouseStyle = document.getElementById("uiHouseStyle");
            $('#uiHouseStyle').empty();
            for (var i in HouseStyle) {
                var opt = new Option(HouseStyle[i]);
                $('#uiHouseStyle').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/RoofStyle";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var RoofStyle = data.RoofStyle;
            var uiRoofStyle = document.getElementById("uiRoofStyle");
            $('#uiRoofStyle').empty();
            for (var i in RoofStyle) {
                var opt = new Option(RoofStyle[i]);
                $('#uiRoofStyle').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/ExterCond";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var ExterCond = data.ExterCond;
            var uiExterCond = document.getElementById("uiExterCond");
            $('#uiExterCond').empty();
            for (var i in ExterCond) {
                var opt = new Option(ExterCond[i]);
                $('#uiExterCond').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Heating";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Heating = data.Heating;
            var uiHeatingMSZoning = document.getElementById("uiHeating");
            $('#uiHeating').empty();
            for (var i in Heating) {
                var opt = new Option(Heating[i]);
                $('#uiHeating').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Electrical";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Electrical = data.Electrical;
            var uiElectrical = document.getElementById("uiElectrical");
            $('#uiElectrical').empty();
            for (var i in Electrical) {
                var opt = new Option(Electrical[i]);
                $('#uiElectrical').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/PoolQC";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var PoolQC = data.PoolQC;
            var uiPoolQC = document.getElementById("uiPoolQC");
            $('#uiPoolQC').empty();
            for (var i in PoolQC) {
                var opt = new Option(PoolQC[i]);
                $('#uiPoolQC').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Utilities";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Utilities = data.Utilities;
            var uiUtilities = document.getElementById("uiUtilities");
            $('#uiUtilities').empty();
            for (var i in Utilities) {
                var opt = new Option(Utilities[i]);
                $('#uiUtilities').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Alley";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Alley = data.Alley;
            var uiAlley = document.getElementById("uiAlley");
            $('#uiAlley').empty();
            for (var i in Alley) {
                var opt = new Option(Alley[i]);
                $('#uiAlley').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Street";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Street = data.Street;
            var uiStreet = document.getElementById("uiStreet");
            $('#uiStreet').empty();
            for (var i in Street) {
                var opt = new Option(Street[i]);
                $('#uiStreet').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Porch";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Porch = data.Porch;
            var uiPorch = document.getElementById("uiPorch");
            $('#uiPorch').empty();
            for (var i in Porch) {
                var opt = new Option(Porch[i]);
                $('#uiPorch').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/Garage";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var Garage = data.Garage;
            var uiGarage = document.getElementById("uiGarage");
            $('#uiGarage').empty();
            for (var i in Garage) {
                var opt = new Option(Garage[i]);
                $('#uiGarage').append(opt);
            }
        }
    });
    console.log("");
    var url = "http://127.0.0.1:5000/CentralAir";
    $.get(url, function(data, status) {
        console.log("");
        if (data) {
            var CentralAir = data.CentralAir;
            var uiCentralAir = document.getElementById("uiCentralAir");
            $('#uiCentralAir').empty();
            for (var i in CentralAir) {
                var opt = new Option(CentralAir[i]);
                $('#uiCentralAir').append(opt);
            }
        }
    });
}

window.onload = onPageLoad;