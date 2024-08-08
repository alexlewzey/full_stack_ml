const btn = document.querySelector("#uploadButton");
btn.onclick = function () {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const base64Image = event.target.result.split(",")[1];

            var xhr = new XMLHttpRequest();
            xhr.open("POST", "/new", true);

            xhr.setRequestHeader("Content-Type", "application/json");

            xhr.onreadystatechange = function () {
                const responseContainer = document.getElementById("result");
                if (xhr.readyState === 4) {
                    if (xhr.status === 200) {
                        var response = JSON.parse(xhr.responseText);
                        console.log(response);
                        responseContainer.innerHTML = `
                            <pre>${JSON.stringify(response, null, 2)}</pre>
                            <img src="data:image/png;base64,${response.image_data}" alt="Uploaded Image" style="width: 500px; height: auto;"/>
                        `;
                    } else {
                        console.error("Error:", xhr.statusText);
                        responseContainer.innerHTML =
                            "<pre>Error raised!</pre>";
                    }
                }
            };

            const payload = {
                image_data: base64Image,
            };

            xhr.send(JSON.stringify(payload));
        };

        reader.readAsDataURL(file);
    } else {
        alert("Please select an image file.");
    }
};
