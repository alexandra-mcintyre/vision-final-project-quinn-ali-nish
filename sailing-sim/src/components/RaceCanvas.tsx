import { useEffect, useRef } from "react";
import { interpolate } from "../utils/interpolate.ts";
import { drawBoat } from "../utils/drawBoat.ts";
import { drawCourse } from "../utils/drawBoat.ts";


export default function RaceCanvas({ data, time }: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw course FIRST (background)
    drawCourse(ctx, data.course);

    // Then boats
    data.boats.forEach((boat: any) => {
      const state = interpolate(boat, time);
      drawBoat(ctx, state, boat.team);
    });

  }, [time]);

  return <canvas ref={canvasRef} width={800} height={600} />;
}