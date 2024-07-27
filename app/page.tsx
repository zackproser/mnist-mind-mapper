'use client'

import React, { useEffect, useRef, useState } from 'react'
import { debounce } from 'lodash'

const isDevelopment = process.env.NODE_ENV === 'development'
const INFERENCE_API_ENDPOINT = isDevelopment ? '/api/predict' : process.env.NEXT_PUBLIC_INFERENCE_API_ENDPOINT

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [lastPoint, setLastPoint] = useState<{ x: number; y: number } | null>(null)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.lineWidth = 25
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        // Set initial background to black
        ctx.fillStyle = 'black'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }
    }
  }, [])

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    const { x, y } = getCoordinates(e)
    setLastPoint({ x, y })
  }

  const stopDrawing = () => {
    setIsDrawing(false)
    setLastPoint(null)
  }

  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (canvas) {
      const rect = canvas.getBoundingClientRect()
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      }
    }
    return { x: 0, y: 0 }
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !lastPoint) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx && canvas) {
      const { x, y } = getCoordinates(e)
      
      const dx = x - lastPoint.x
      const dy = y - lastPoint.y
      const distance = Math.sqrt(dx * dx + dy * dy)
      
      // Adjust density based on drawing speed
      const density = Math.min(1, 0.5 + 30 / distance)
      
      // Create a radial gradient for grayscale effect (inverted)
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, ctx.lineWidth / 2)
      gradient.addColorStop(0, `rgba(255, 255, 255, ${density})`)  // Dense center (white)
      gradient.addColorStop(1, 'rgba(255, 255, 255, 0)')    // Transparent edges
      
      ctx.strokeStyle = gradient
      
      ctx.beginPath()
      ctx.moveTo(lastPoint.x, lastPoint.y)
      ctx.lineTo(x, y)
      ctx.stroke()
      
      setLastPoint({ x, y })
    }
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx && canvas) {
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
    setPrediction(null)
  }

  const getPrediction = debounce(async () => {
    const canvas = canvasRef.current
    if (canvas) {
      const imageData = canvas.toDataURL('image/png')
      
      // Log the base64 encoded image data
      console.log('Base64 encoded image data:', imageData)
      
      if (!INFERENCE_API_ENDPOINT) {
        console.error('Inference API endpoint is not defined')
        return
      }

      try {
        const response = await fetch(INFERENCE_API_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData }),
        })
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const data = await response.json()
        setPrediction(data.prediction)
      } catch (error) {
        console.error('Error getting prediction:', error)
        setPrediction(null)
      }
    }
  }, 150)

  useEffect(() => {
    getPrediction()
  }, [isDrawing])

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 bg-black text-white font-mono">
      <h1 className="text-6xl font-extrabold mb-12 text-green-500 transform -skew-x-6">DRAW A NUMBER BETWEEN 1-10</h1>
      <div className="mb-8 text-3xl border-4 border-yellow-400 p-4 transform skew-x-3">
        Prediction: <span className="text-yellow-400">{prediction !== null ? prediction : 'N/A'}</span>
      </div>
      <div className="relative">
        <div className="absolute inset-0 bg-red-500 transform translate-x-2 translate-y-2"></div>
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="relative border-4 border-white bg-black"
          style={{ backgroundColor: 'black' }} // Ensure black background
          onMouseDown={startDrawing}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
          onMouseMove={draw}
        />
      </div>
      <button
        onClick={clearCanvas}
        className="mt-12 px-8 py-4 bg-white text-black text-2xl font-bold uppercase tracking-widest hover:bg-red-500 hover:text-white transition-colors duration-300 transform hover:scale-105"
      >
        Clear
      </button>
      <div className="mt-12 text-xl border-4 border-green-400 p-4 transform -skew-x-3">
        Created By <a href="https://zackproser.com" target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-red-500 transition-colors duration-300">Zachary Proser</a>
      </div>
      <button
        onClick={() => setShowExplanation(!showExplanation)}
        className="mt-6 px-6 py-3 bg-green-400 text-black text-xl font-bold uppercase tracking-wider hover:bg-red-500 hover:text-white transition-colors duration-300 transform hover:scale-105"
      >
        What is this?
      </button>
      {showExplanation && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 border-4 border-green-400 p-6 max-w-2xl relative">
            <button
              onClick={() => setShowExplanation(false)}
              className="absolute top-2 right-2 text-green-400 hover:text-red-500 text-2xl font-bold"
            >
              ×
            </button>
            <h2 className="text-2xl font-bold mb-4 text-green-400">About This Project</h2>
            <p className="mb-2">I built this project to better understand neural networks. It uses PyTorch and the MNIST dataset, which is a large collection of handwritten digits commonly used for training image processing systems.</p>
            <p className="mb-2">The model was trained to recognize hand-drawn images and then wrapped in a Flask server. The frontend is built with Next.js and deployed to Vercel.</p>
            <p>When a user draws a number, the frontend sends the image data to the Flask backend. The backend processes the image using the trained PyTorch model and returns a prediction. The frontend then displays this prediction to the user in real-time.</p>
          </div>
        </div>
      )}
    </main>
  )
}
