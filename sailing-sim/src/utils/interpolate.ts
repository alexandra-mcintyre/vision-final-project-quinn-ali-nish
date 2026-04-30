export function interpolate(boat: any, time: number) {
  const i = Math.floor(time);
  const alpha = time - i;

  if (i >= boat.positions.length - 1) {
    return {
      x: boat.positions.at(-1)[0],
      y: boat.positions.at(-1)[1],
      heading: boat.heading.at(-1)
    };
  }

  const [x1, y1] = boat.positions[i];
  const [x2, y2] = boat.positions[i + 1];

  return {
    x: (1 - alpha) * x1 + alpha * x2,
    y: (1 - alpha) * y1 + alpha * y2,
    heading: (1 - alpha) * boat.heading[i] + alpha * boat.heading[i + 1]
  };
}