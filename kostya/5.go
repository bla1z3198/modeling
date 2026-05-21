package main

import (
	"fmt"
	"math"
	"time"
)

type UAVState struct {
	Velocity float64 // V
	Height   float64 // h
	Pitch    float64 // theta
	Heading  float64 // psi
	Roll     float64 // phi
}

type PID struct {
	Kp float64
	Ki float64
	Kd float64

	Integral  float64
	PrevError float64
}

func (p *PID) Update(err, dt float64) float64 {
	p.Integral += err * dt

	derivative := (err - p.PrevError) / dt

	output := p.Kp*err +
		p.Ki*p.Integral +
		p.Kd*derivative

	p.PrevError = err

	return output
}

// Линейная модель МБЛА
func UpdateState(
	s *UAVState,
	elevator float64,
	throttle float64,
	aileron float64,
	dt float64,
) {
	// Ограничения
	elevator = clamp(elevator, -0.3, 0.3)
	aileron = clamp(aileron, -0.3, 0.3)
	throttle = clamp(throttle, -1.0, 1.0)

	// Тангаж
	s.Pitch += elevator * dt

	// Крен
	s.Roll += aileron * dt

	// Высота
	s.Height += s.Velocity * s.Pitch * dt

	// Курс
	s.Heading += s.Roll * dt

	// Скорость
	s.Velocity += throttle * dt

	// Демпфирование
	s.Pitch *= 0.999
	s.Roll *= 0.999
	s.Velocity *= 0.999
}

func clamp(v, min, max float64) float64 {
	return math.Max(min, math.Min(max, v))
}

func main() {

	dt := 0.01
	simTime := 60.0

	// Начальное состояние
	state := UAVState{
		Velocity: 20,
		Height:   100,
		Pitch:    0,
		Heading:  0,
		Roll:     0,
	}

	// Целевые значения
	targetHeight := 300.0
	targetVelocity := 30.0
	targetHeading := 45.0

	// PID-регуляторы

	// Высота
	altitudePID := PID{
		Kp: 0.015,
		Ki: 0.0001,
		Kd: 0.01,
	}

	// Скорость
	speedPID := PID{
		Kp: 0.5,
		Ki: 0.01,
		Kd: 0.1,
	}

	// Курс
	headingPID := PID{
		Kp: 0.02,
		Ki: 0.0001,
		Kd: 0.01,
	}

	fmt.Println("время | высота | скорость | курс")

	for t := 0.0; t <= simTime; t += dt {

		// Ошибки
		heightError := targetHeight - state.Height
		speedError := targetVelocity - state.Velocity
		headingError := targetHeading - state.Heading

		// PID управление
		elevator := altitudePID.Update(heightError, dt)
		throttle := speedPID.Update(speedError, dt)
		aileron := headingPID.Update(headingError, dt)

		// Обновление состояния
		UpdateState(
			&state,
			elevator,
			throttle,
			aileron,
			dt,
		)

		// Вывод каждые 0.5 сек
		if int(t*100)%50 == 0 {
			time.Sleep(time.Millisecond * 500)
			fmt.Printf(
				"%.2f|   %.2f|   %.2f|   %.2f|\n",
				t,
				state.Height,
				state.Velocity,
				state.Heading,
			)
		}
	}

	fmt.Println("\n=== Финальное состояние ===")
	fmt.Printf("Высота: %.2f м\n", state.Height)
	fmt.Printf("Скорость: %.2f м/с\n", state.Velocity)
	fmt.Printf("Курс: %.2f град\n", state.Heading)
}
