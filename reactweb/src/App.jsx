import { useEffect, useState } from "react";

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

    <div
      style={{
        padding: "20px",
        fontFamily: "Arial",
      }}
    >

      <h1 style={{color : "black"}}>Texture Clothing GAN</h1>



      {/* ---------------------------- */}
      {/* IMAGE */}
      {/* ---------------------------- */}

      <div>

        <p>Upload Image</p>

        <input
          type="file"
          onChange={handleImageChange}
        />

      </div>



      <br />



      {/* ---------------------------- */}
      {/* CLASS */}
      {/* ---------------------------- */}

      <div>

        <p>Select Clothing Class</p>

        <select
          value={selectedClass}
          onChange={(event) => {

            setSelectedClass(event.target.value);
          }}
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



      <br />



      {/* ---------------------------- */}
      {/* GENERATE */}
      {/* ---------------------------- */}

      <button onClick={handleGenerate}>
        Generate
      </button>



      <br />
      <br />



      {/* ---------------------------- */}
      {/* SERVER RESPONSE */}
      {/* ---------------------------- */}

      {serverResponse && (

        <div>

          <h2>Server Response</h2>



          {/* IMAGE */}

          <img
            src={serverResponse.result.generated_image.url}
            alt="Generated"
            width="300"
          />



          <br />
          <br />



          {/* RESPONSE DETAILS */}

          <div>

            <p>
              <strong>Status:</strong>
              {" "}
              {serverResponse.status}
            </p>

            <p>
              <strong>Request ID:</strong>
              {" "}
              {serverResponse.request_metadata.request_id}
            </p>

            <p>
              <strong>Selected Class:</strong>
              {" "}
              {serverResponse.user_input.selected_class}
            </p>

            <p>
              <strong>Saved Upload:</strong>
              {" "}
              {serverResponse.upload.saved_filename}
            </p>

            <p>
              <strong>Model:</strong>
              {" "}
              {serverResponse.model_info.model_name}
            </p>

            <p>
              <strong>Processing Time:</strong>
              {" "}
              {serverResponse.performance.processing_time_seconds}
              {" "}seconds
            </p>

          </div>

        </div>

      )}

    </div>
  );
}

export default App;