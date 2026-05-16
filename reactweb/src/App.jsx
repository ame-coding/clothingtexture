import { useEffect, useState } from "react";
import "./App.css";

function App() {

  // -----------------------------------
  // SERVER DATA
  // -----------------------------------

  const [classes, setClasses] = useState([]);




  // -----------------------------------
  // USER INPUTS
  // -----------------------------------

  const [selectedClass, setSelectedClass] = useState("");

  const [selectedImage, setSelectedImage] = useState(null);




  // -----------------------------------
  // SERVER RESPONSE
  // -----------------------------------

  const [serverResponse, setServerResponse] = useState(null);




  // -----------------------------------
  // LOAD CLASSES
  // -----------------------------------

  useEffect(() => {

    async function loadClasses() {

      try {

        const response = await fetch(
          "http://localhost:8000/classes"
        );

        const data = await response.json();

        setClasses(data.classes);

      } catch (error) {

        console.log(error);
      }
    }

    loadClasses();

  }, []);




  // -----------------------------------
  // HANDLE IMAGE
  // -----------------------------------

  function handleImageChange(event) {

    const file = event.target.files[0];

    setSelectedImage(file);
  }




  // -----------------------------------
  // GENERATE REQUEST
  // -----------------------------------

  async function handleGenerate() {

    // Validate image
    if (!selectedImage) {

      alert("Please upload an image");

      return;
    }


    // Validate class
    if (!selectedClass) {

      alert("Please select a clothing class");

      return;
    }




    // -----------------------------------
    // STRUCTURED REQUEST OBJECT
    // -----------------------------------

    const requestData = {

      request_metadata: {

        request_id: crypto.randomUUID(),

        timestamp: new Date().toISOString(),

        client_version: "1.0",

        request_type: "clothing_generation",
      },



      user_input: {

        selected_class: selectedClass,
      },



      generation_settings: {

        mode: "test",

        quality: "standard",

        generation_steps: 1,
      },



      model_settings: {

        model_name: "ConditionalAI-Test",

        device: "cpu",
      },



      output_settings: {

        return_preview: true,

        save_output: true,
      }
    };




    // -----------------------------------
    // CREATE FORM DATA
    // -----------------------------------

    const formData = new FormData();

    formData.append("image", selectedImage);

    formData.append(
      "request_data",
      JSON.stringify(requestData)
    );




    // -----------------------------------
    // SEND REQUEST
    // -----------------------------------

    try {

      const response = await fetch(
        "http://localhost:8000/generate",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      setServerResponse(data);

    } catch (error) {

      console.log(error);
    }
  }




  // -----------------------------------
  // UI
  // -----------------------------------

  return (
  <div className="app-container">

    <div className="main-card">

      <h1 className="title">
        Texture Clothing GAN
      </h1>

    {/* IMAGE */}

    <div className="section">

      <label className="label">
        Upload Texture Image
      </label>

      <label className="upload-box">

        <input
          type="file"
          onChange={handleImageChange}
          className="hidden-input"
        />

        <span className="upload-text">
          Choose Image
        </span>

      </label>

      {selectedImage && (

        <p className="file-name">
          Selected File:
          {" "}
          <strong>{selectedImage.name}</strong>
        </p>

      )}

    </div>

      {/* CLASS */}

      <div className="section">

        <label className="label">
          Select Clothing Class
        </label>

        <select
          value={selectedClass}
          onChange={(event) => {
            setSelectedClass(event.target.value);
          }}
          className="select-input"
        >

          <option value="">
            Choose One
          </option>

          {classes.map((item) => (

            <option
              key={item}
              value={item}
            >
              {item}
            </option>

          ))}

        </select>

      </div>

      {/* BUTTON */}

      <button
        onClick={handleGenerate}
        className="generate-button"
      >
        Generate Clothing
      </button>

      {/* SERVER RESPONSE */}

      {serverResponse && (

        <div className="response-container">

          <h2 className="response-title">
            Generated Result
          </h2>

          <img
            src={serverResponse.result.generated_image.url}
            alt="Generated"
            className="generated-image"
          />

          <div className="info-grid">

            <div className="info-card">
              <div className="info-title">
                Status
              </div>

              <div className="info-value">
                {serverResponse.status}
              </div>
            </div>

            <div className="info-card">
              <div className="info-title">
                Request ID
              </div>

              <div className="info-value">
                {serverResponse.request_metadata.request_id}
              </div>
            </div>

            <div className="info-card">
              <div className="info-title">
                Selected Class
              </div>

              <div className="info-value">
                {serverResponse.user_input.selected_class}
              </div>
            </div>

            <div className="info-card">
              <div className="info-title">
                Saved Upload
              </div>

              <div className="info-value">
                {serverResponse.upload.saved_filename}
              </div>
            </div>

            <div className="info-card">
              <div className="info-title">
                Model
              </div>

              <div className="info-value">
                {serverResponse.model_info.model_name}
              </div>
            </div>

            <div className="info-card">
              <div className="info-title">
                Processing Time
              </div>

              <div className="info-value">
                {serverResponse.performance.processing_time_seconds}
                {" "}seconds
              </div>
            </div>

          </div>

        </div>

      )}

    </div>

  </div>
);
}

export default App;