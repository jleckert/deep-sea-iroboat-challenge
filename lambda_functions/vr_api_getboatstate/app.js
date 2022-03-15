var http = require("http");
const fs = require("fs");
let rawdata = fs.readFileSync("data.json");
let vrUsers = JSON.parse(rawdata);
let allUrls = [];

exports.handler = function (event, context) {
  let api_url = process.env.api_url;

  console.log("**start request to **" + api_url);
  console.log("vrUsers", vrUsers);

  vrUsers.users.forEach((user) => {
    user.races.forEach((race) => {
      let url = api_url + "/" + race + "/" + user.id;
      allUrls.push(url);
    });
  });

  function call_vr_api(url) {
    console.log("** URL", url);

    return new Promise(function (resolve, reject) {
      http
        .get(url, function (res) {
          console.log("**Got response ** : " + res.statusCode);
          resolve(res.statusCode);
        })
        .on("error", function (e) {
          console.log("Got error: " + e.message);
          reject(e.message);
        });
    });
  }

  async function processsUsers() {
    for (var i = 0; i < allUrls.length; i++) {
      await call_vr_api(allUrls[i]);
    }
  }

  processsUsers();
};
