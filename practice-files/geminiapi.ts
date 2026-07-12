//http raw gemini api call using typescript.

import { FunctionResponseScheduling, GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";

dotenv.config();

async function main() {
  // initializing the client.
  const ai = new GoogleGenAI({
    apiKey: process.env.GOOGLE_API_KEY
  });

  try {
    // Calling the 2026 "Workhorse" model
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-lite",
      contents: "Explain what are the tensors in no more than 3 sentences for a beginner."
    });

    // log the responses

    console.log("TS Gemini 3.1 says:", response.text);

  } catch (error: unknown) {
   
    if (error instanceof Error) {
      console.error("Connection failed:", error.message);
    } else {
      console.error("An unexpected error occurred:", error);
    }
  }
}

(async () => {
  try {
    await main();
    console.log("--- Execution Complete ---");
  } catch (err) {
    if (err instanceof Error) {
      console.error("Fatal Script Error:", err.message);
    }
    process.exit(1);
  }
})();