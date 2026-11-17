import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/newsreader/400.css";
import "@fontsource/newsreader/400-italic.css";
import "@fontsource/newsreader/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "./styles.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
