import { useEffect, useRef } from 'react'

export function ParticleLayer({ pulse }: { pulse: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const context = canvas.getContext('2d')
    if (!context) return
    const particles = Array.from({ length: Math.min(12 + pulse * 2, 42) }, (_, index) => ({
      x: -20 - index * 19,
      yRatio: 0.25 + ((index * 37) % 50) / 100,
      radius: index % 4 === 0 ? 2.8 : 1.4,
      speed: 42 + (index % 5) * 9,
      color: index % 5 === 0 ? '#42d9ff' : '#91ff4a'
    }))
    let width = 0
    let height = 0
    let animation = 0
    let previous = performance.now()
    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      width = canvas.clientWidth
      height = canvas.clientHeight
      canvas.width = Math.round(width * ratio)
      canvas.height = Math.round(height * ratio)
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    resize()
    const draw = (now: number) => {
      const delta = Math.min((now - previous) / 1000, 0.05)
      previous = now
      context.clearRect(0, 0, width, height)
      for (const [index, particle] of particles.entries()) {
        particle.x = (particle.x + particle.speed * delta) % (width + 80)
        context.globalAlpha = 0.25 + 0.65 * Math.sin((particle.x + index * 17) / 90) ** 2
        context.fillStyle = particle.color
        context.beginPath()
        context.arc(particle.x, height * particle.yRatio, particle.radius, 0, Math.PI * 2)
        context.fill()
      }
      animation = requestAnimationFrame(draw)
    }
    animation = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(animation)
      observer.disconnect()
    }
  }, [pulse])

  return <canvas ref={canvasRef} className="particle-layer" aria-hidden="true" />
}
