# Neural Network Digit Recognizer

A web application that uses a neural network to recognize hand-drawn digits between 1 and 10.

## Overview

This project combines a PyTorch-based neural network with a Next.js frontend to create an interactive digit recognition application. Users can draw a digit on a canvas, and the application will predict the drawn number in real-time.

## Features

- Interactive canvas for drawing digits
- Real-time prediction using a trained neural network
- Brutalist design aesthetic
- Responsive layout for both desktop and mobile devices
- Informative modal explaining the project's purpose and technology stack

## Technology Stack

- Frontend: Next.js, React, TypeScript
- Styling: Tailwind CSS
- Backend: Flask
- Machine Learning: PyTorch
- Dataset: MNIST

## How It Works

1. The user draws a digit on the canvas.
2. The frontend sends the image data to the Flask backend.
3. The backend processes the image using a trained PyTorch model.
4. The prediction is sent back to the frontend and displayed in real-time.

## Getting Started

### Prerequisites

- Node.js
- Python 3.x
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/neural-network-digit-recognizer.git
   cd neural-network-digit-recognizer
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Install backend dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the development server:
   ```
   npm run dev
   ```

2. Open your browser and navigate to `http://localhost:3000`

## Deployment

This application is designed to be deployed on Vercel. Follow the Vercel documentation for deploying Next.js applications with API routes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- MNIST dataset for providing the training data
- Next.js and Flask for providing robust frameworks
- Vercel for hosting and deployment
