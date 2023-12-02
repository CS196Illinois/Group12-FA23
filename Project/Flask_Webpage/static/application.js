const countriesString = "Zambia,DR Congo,Egypt,Greece,Bolivia,Chile,Croatia,Serbia,Italy,Belgium,Wales,Northern Ireland,Austria,Switzerland,Guyana,Trinidad and Tobago,Argentina,Paraguay,Spain,Portugal,Uruguay,Scotland,England,Netherlands,South Korea,Taiwan,Hungary,Czechoslovakia,Albania,Poland,Brazil,Republic of Ireland,Romania,France,Ghana,Nigeria,Turkey,Iran,Yugoslavia,Denmark,Bulgaria,Sweden,Finland,Zimbabwe,Australia,Norway,Mexico,South Africa,United States,Madagascar,Mauritius,Luxembourg,Suriname,El Salvador,Guatemala,Tanzania,Zanzibar,Kenya,Uganda,Jamaica,Haiti,Afghanistan,Israel,Curacao,Germany,Guadeloupe,Costa Rica,Nicaragua,Panama,Barbados,Grenada,Martinique,Iceland,New Caledonia,New Zealand,Vanuatu,Fiji,Venezuela,Aruba,Myanmar,India,Pakistan,Peru,Sri Lanka,German DR,Russia,China PR,Tahiti,Gambia,Sierra Leone,Ecuador,Honduras,Hong Kong,Singapore,Malaysia,Indonesia,Guinea-Bissau,Philippines,Burundi,Japan,Ethiopia,Djibouti,Cambodia,Thailand,Lebanon,Cuba,Vietnam Republic,Belarus,Kyrgyzstan,Estonia,Lithuania,Moldova,Togo,North Korea,Sudan,Malta,Syria,Colombia,Tunisia,Libya,Canada,Malawi,Morocco,Benin,Cape Verde,Cameroon,Central African Republic,Mali,Gabon,Burkina Faso,Ivory Coast,Congo,Iraq,Cyprus,Saint Lucia,Dominica,Saint Vincent and the Grenadines,Senegal,Guinea,French Guiana,Puerto Rico,Algeria,Kuwait,Jordan,Papua New Guinea,Solomon Islands,Liberia,Somalia,Laos,Saudi Arabia,Chad,Bermuda,Niger,Montenegro,Palestine,Yemen,Bahrain,Oman,Mauritania,Eswatini,Botswana,Qatar,Lesotho,Macau,United Arab Emirates,Faroe Islands,Saint Kitts and Nevis,Nepal,Antigua and Barbuda,Bangladesh,Seychelles,Equatorial Guinea,Mozambique,Guam,Angola,Dominican Republic,Rwanda,Armenia,Georgia,Latvia,Azerbaijan,Ukraine,Kazakhstan,Liechtenstein,Cayman Islands,Namibia,British Virgin Islands,San Marino,Slovenia,Turkmenistan,Tajikistan,Uzbekistan,Slovakia,Czech Republic,Vietnam,North Macedonia,Bosnia and Herzegovina,Maldives,Andorra,Gibraltar"
const countries = countriesString.split(",")
var homeMenu;
var awayMenu;
var homeInput;
var awayInput;
var homeSelected = false;
var awaySelected = false;
countries.sort();

function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}

function filterFunction(input, menu) {
    var term = input.value

    removeAllChildNodes(menu)

    for (let i in countries) {
        if (countries[i] == term) {
            return true;
        }

        if (countries[i].toLowerCase().indexOf(term.toLowerCase()) >= 0) {
            const p = document.createElement("p");
            p.classList = ["h5"]
            p.textContent = countries[i]
            p.onclick = () => {
                input.value = countries[i]
                filterFunction(input, menu)
            }

            menu.appendChild(p)
        }
    }

    return false;
}

function filterHome() {
    filterFunction(homeInput, homeMenu)
}

function filterAway() {
    filterFunction(awayInput, awayMenu)
}

function showError() {

}


// Initialize
onload = () => {
    homeMenu = document.getElementById("homeMenu")
    awayMenu = document.getElementById("awayMenu")
    homeInput = document.getElementById("homeInput")
    awayInput = document.getElementById("awayInput")

    filterHome()
    filterAway()

    document.getElementById("predict").onclick = () => {
        let home = homeInput.value;
        let away = awayInput.value;
        let year = document.getElementById("year").value;
        let checked = document.querySelector('input[name = "field"]:checked');
        
        if (countriesString.indexOf(home) == -1 || countriesString.indexOf(away) == -1 || year < 1950 || year > 2024 || checked == null || home == away) {
            alert("Invalid Input");
        } else {
            window.location.replace("/application/" + home + "/" + away + "/" + year + "/" + checked.value)
        }
    }
}