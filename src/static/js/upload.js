
const uploadForm = document.getElementById('uploadForm')
const fileInput = document.getElementById('fileInput')
const responseContainer = document.getElementById('response-container')

uploadForm.addEventListener("submit", async (event)=>{
  event.preventDefault();

  const files = fileInput.files;

  if (!fileInput.files[0]){
    alert("Please select a file to upload.")
    return;
  }

  const formData = new FormData();
  formData.append('file', files[0]);

  try {
    const response = await fetch("/upload",{
      method: 'POST',
      body: formData,
    });

    if (!response.ok){
      throw new Error(`Server error: ${response.status}`)
    }

    const result = await response.json() // Assuming a JSON is returned
    responseContainer.textContent = `Upload successful: ${result.filename}`
    } catch(error){
      responseContainer.textContent = `Upload failed: ${error.message}`
    }
});
