import { useEffect, useState, useRef } from "react";
import RaceCanvas from "./components/RaceCanvas";
import { raceData } from "./data/raceData";

export default function App() {
  const [time, setTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const isPlayingRef = useRef(isPlaying);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    let animationId: number;
    let lastTime = performance.now();

    function animate(now: number) {
      const delta = (now - lastTime) / 1000;
      lastTime = now;

      if (isPlayingRef.current) {
        setTime(prev => prev + delta);
      }

      animationId = requestAnimationFrame(animate);
    }
    animationId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationId);
  }, []);

  return (
      <>
        <h1>Team Race Simulator</h1>

        <RaceCanvas data={raceData} time={time}/>

        <div>
          <button onClick={() => setIsPlaying(prev => !prev)}>
            {isPlaying ? "Pause" : "Play"}
          </button>

          <button onClick={() => {
            setTime(0);
            setIsPlaying(false);
          }}>
            Restart
          </button>
        </div>
      </>
  );
}