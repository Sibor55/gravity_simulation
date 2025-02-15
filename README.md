## Requirements

- Python 3.x
- `numpy` library

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/physics-simulation.git
   cd physics-simulation
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy
   ```

3. Run the simulation:
   ```bash
   python main.py
   ```

## Usage

The simulation starts with a predefined set of circles. You can modify the initial conditions in the `main()` function to create your own scenarios. For example:

- Change the number of circles.
- Adjust the masses, radii, and initial velocities.
- Add or remove gravitational objects.

### Example Configuration

```python
circle1 = Circle(7, 10, 25, fill=8, vx=7, vy=0, mass=8)
circle2 = Circle(2, 40, 30, fill=3, mass=3)
circle3 = Circle(4, 20, 20, fill=5, mass=5)
gravity = Circle(1, WIDTH // 2, HEIGHT + 40, 0, GRAVITY, 0, 0, False, False)
circles = [gravity, circle1, circle2, circle3]
```

## Controls

- The simulation runs in an infinite loop. To stop it, press `Ctrl+C`.
- Adjust the `time.sleep(0.05)` value in the `main()` function to control the speed of the simulation.


![изображение](https://github.com/user-attachments/assets/91d6382b-d630-416d-8ce1-42e8dac58e6b)
