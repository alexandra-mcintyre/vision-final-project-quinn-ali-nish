type Course = {
  startLine: [[number, number], [number, number]];
  finishLine: [[number, number], [number, number]];
  marks: {
    m1: [number, number];
    m2: [number, number];
    m3: [number, number];
    m4: [number, number];
  };
};

type Boat = {
  id: string;
  team: "A" | "B";
  positions: [number, number][];
  heading: number[];
};

type RaceData = {
  course: Course;
  boats: Boat[];
};

export function drawCourse(ctx: CanvasRenderingContext2D, course: any) {
  ctx.strokeStyle = "black";
  ctx.lineWidth = 2;

  // Start line
  drawLine(ctx, course.startLine, "green");

  // Finish line
  drawLine(ctx, course.finishLine, "red");

  // Marks
  Object.values(course.marks).forEach((mark: any) => {
    drawMark(ctx, mark);
  });
}

function drawLine(ctx: any, [[x1, y1], [x2, y2]]: any, color: string) {
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function drawMark(ctx: any, [x, y]: any) {
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fillStyle = "orange";
  ctx.fill();
}

export function drawBoat(ctx: any, boat: any, team: string) {
  ctx.save();
  ctx.translate(boat.x, boat.y);
  ctx.rotate(boat.heading);

  ctx.beginPath();
  ctx.moveTo(12, 0);
  ctx.lineTo(-10, 6);
  ctx.lineTo(-10, -6);
  ctx.closePath();

  ctx.fillStyle = team === "A" ? "blue" : "red";
  ctx.fill();

  ctx.restore();
}