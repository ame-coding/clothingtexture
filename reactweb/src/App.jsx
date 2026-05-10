import { useState } from "react";

function App() {

  const [image, setImage] = useState(null);

  const generateImage = async () => {

    const response = await fetch("http://localhost:8000/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        hair_color: "blue",
        eye_color: "green",
        style: "anime",
      }),
    });

    const data = await response.json();

    setImage(data.image_url);
  };

  return (
    <div style={{ padding: "20px" }}>
      <button onClick={generateImage}>
        Generate
      </button>

      {image && (
        <div>
          <img
            src={image}
            alt="Generated"
            style={{ width: "512px" }}
          />
        </div>
      )}
    </div>
  );
}

export default App;