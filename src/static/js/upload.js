const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const responseContainer = document.getElementById("response-container");

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const files = fileInput.files;

  if (!fileInput.files[0]) {
    alert("Please select a file to upload.");
    return;
  }

  const formData = new FormData();
  formData.append("file", files[0]);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    // Log the raw response to ensure it's received correctly
    console.log("Raw response:", response);

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json(); // Parse the JSON response
    console.log("Parsed response:", result);

    // Correctly display the result
    if (result.result) {
      responseContainer.textContent = `Upload successful:\n${result.result}`;
      // } else {
      //   responseContainer.textContent = `Upload successful, but no detailed result: ${result.filename}`;
    }
  } catch (error) {
    // Handle and log any errors
    console.error("Error during fetch:", error);
    responseContainer.textContent = `Upload failed: ${error.message}`;
  }
});
