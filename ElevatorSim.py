import random
import math
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Constants

MEAN_TIME_BETWEEN_REQUESTS = 120

AVG_TIME_ON_SECONDARY_FLOOR = 30
SIGMA_TIME_ON_SECONDARY_FLOOR = 60

AVG_TIME_IN_SYSTEM = 240 # 4 hours
SIGMA_TIME_IN_SYSTEM = 60

def exponential(mean):
    return -mean*math.log(random.random())

def normal(mean, sigma):
    r1 = random.random()
    r2 = random.random()
    return_normal_1 = (math.sqrt(-2*math.log(r1))*math.cos(2*math.pi*r2)) * sigma + mean
    return_normal_2 = (math.sqrt(-2*math.log(r1))*math.sin(2*math.pi*r2)) * sigma + mean
    if return_normal_1 > 0:
      return return_normal_1
    elif return_normal_2 > 0:
      return return_normal_2
    else:
      return normal(mean, sigma)

class Person:
    def __init__(self, arrival_time, departure_time, non_lobby_floors):
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.primary_floor = random.choice(non_lobby_floors)
        self.current_floor = 0 #people always start in lobby
        self.wait_time_list = []
        self.on_elevator = False
        self.non_lobby_floors = non_lobby_floors

    def generate_initial_request(self):
        request = Request(self)
        request.time_of_request = self.arrival_time
        request.departure_floor = 0
        request.arrival_floor =  self.primary_floor
        return request

    def generate_next_request(self, clock):
        if self.current_floor == self.primary_floor:
            #generate next time this person needs to visit a non-lobby floor
            #exp dist
            request = Request(self)
            request.departure_floor = self.current_floor

            request_time = exponential(MEAN_TIME_BETWEEN_REQUESTS) + clock

            if request_time > self.departure_time:
                request.arrival_floor = 0 #Lobby
                request.time_of_request = max(clock, self.departure_time)
                return request

            request.time_of_request = request_time
            floor_options = self.non_lobby_floors.copy()
            floor_options.remove(self.current_floor)
            request.arrival_floor = random.choice(floor_options)
            return request
        else:
            #genernate the time when they will return to their primary floor
            #normal dist
            request = Request(self)
            request.time_of_request = normal(AVG_TIME_ON_SECONDARY_FLOOR,SIGMA_TIME_ON_SECONDARY_FLOOR) + clock
            request.departure_floor = self.current_floor
            request.arrival_floor = self.primary_floor
            return request


class Request:
    def __init__(self, person):
        self.time_of_request = None
        self.departure_floor = None
        self.arrival_floor = None
        self.person = person

    def get_time(self):
        return self.time_of_request

    def compare_to(self, cmp_event):
        cmp_time = cmp_event.get_time()
        if self.time_of_request < cmp_time:
            return -1
        if self.time_of_request == cmp_time:
            return 0
        return 1

# Essentially the first half of a request. Used in Elevator class.
class PickUp:
    def __init__(self, person, floor, dir):
        self.person = person
        self.floor = floor
        self.direction = dir

# Essentially the second half of a request. Used in Elevator class.
class DropOff:
    def __init__(self, person, floor, dir):
        self.person = person
        self.floor = floor
        self.direction = dir

# Represents a group of simultaneous actions (PickUps and DropOffs) to be performed on a floor.
class FloorBatchAction:
    def __init__(self, actions):
        self.actions = actions
        self.floor = actions[0].floor
        self.direction = actions[0].direction

    def add_action(self, action):
         self.actions.append(action)

    # Returns set of all people involved in the FloorBatchAction
    def get_people(self):
         return {action.person for action in self.actions}


class Wait:
    TOLOBBY = -1
    INTERFLOOR = 0
    FROMLOBBY = 1

    def __init__(self, req):
        self.start_time = req.get_time()
        self.journey_type = None
        if req.arrival_floor == 0:
            self.journey_type = Wait.TOLOBBY
        elif req.departure_floor == 0:
            self.journey_type = Wait.FROMLOBBY
        else:
            self.journey_type = Wait.INTERFLOOR
        self.total_wait_time = 0
        self.hall_wait_time = 0
        self.travel_wait_time = 0

        


class Elevator:

    velocity = 30   # floors/min, based on typical pasenger elevator speeds of
                    # 300 ft/min and typical commercial floor height of 10ft.
    stopping_time = 10/60   # Elevator stops at a floor for a constant 10 seconds.

    # Constants to represent direction of travel.
    IDLE = 0
    UP = 1
    DOWN = -1

    def __init__(self, building):
      self.direction = Elevator.IDLE
      self.position = 0
      # List of FloorBatchActions, ordered as per elevator algorithm.
      self.fbas = []
      self.next_stop = None
      self.clock = 0
      self.wait_time_remaining = Elevator.stopping_time
      self.building = building

    # Time to travel directly from start to dest
    def travel_time(start, dest):
      return abs(dest-start)/Elevator.velocity

    # Time until elevator makes next stop, provided self.next_stop != None
    def get_next_stop_time(self):
      return self.clock + self.wait_time_remaining + Elevator.travel_time(self.position, self.next_stop.floor)

    # Add time to total_wait_time of all people in FloorBatchAction list.
    def add_wait_to_all(self, time):
      people = set()
      for fba in self.fbas:
          people = people.union(fba.get_people())
      for person in people:
          person.wait_time_list[-1].total_wait_time += time
          if person.on_elevator:
              person.wait_time_list[-1].travel_wait_time += time
          else:
              person.wait_time_list[-1].hall_wait_time += time

    # Update this elevator to the point in time when its next stop occurs.
    def update_to_next_stop(self, is_copy=False):
      next_stop_time = self.get_next_stop_time()
      added_wait = next_stop_time - self.clock
      self.add_wait_to_all(added_wait)
      self.clock = next_stop_time
      self.position = self.next_stop.floor
      self.wait_time_remaining = Elevator.stopping_time

      # is_copy=True is used in wait-time forecasting for destination dispatch.
      if not is_copy:
          for action in self.next_stop.actions:
              if type(action) == DropOff:
                  action.person.on_elevator = False
                  action.person.current_floor = action.floor
                  if action.floor != 0:
                      self.building.fel.enqueue(action.person.generate_next_request(self.clock))
              else:
                  action.person.on_elevator = True

      # Dequeue first two FBAs if they correspond to the same floor (this only occurs when
      # the elevator is switching directions, since actions are grouped by floor and direction).
      if len(self.fbas) > 1 and self.fbas[0].floor==self.fbas[1].floor:
        for action in self.fbas[1].actions:
            action.person.on_elevator = True
        self.fbas = self.fbas[2:]
      else:
        self.fbas = self.fbas[1:]

      if len(self.fbas) == 0:
            self.next_stop = None
            self.direction = Elevator.IDLE
      else:
          self.next_stop = self.fbas[0]
          self.direction = (self.next_stop.floor-self.position)/abs(self.next_stop.floor-self.position)


    # This method may only be called when the elevator has no stops between now and time t.
    def update_to_time_t(self, t):
      differential = t - self.clock
      self.add_wait_to_all(differential)
      self.clock = t
      if self.direction != Elevator.IDLE:
          if self.wait_time_remaining - differential >= 0:
              self.wait_time_remaining -= differential
          else:
              differential -= self.wait_time_remaining
              self.wait_time_remaining = 0
              self.position += self.direction * Elevator.velocity * differential


    # Distance elevator must travel, given its current workload, to service a new request.
    def compute_distance(self, req):
      req_dir = (req.arrival_floor-req.departure_floor)/abs(req.arrival_floor-req.departure_floor)

      if self.direction == Elevator.IDLE:
          return abs(req.departure_floor - self.position)
      elif self.direction == Elevator.UP:
          if req_dir == Elevator.UP:
              if req.departure_floor >= self.position:
                    return abs(req.departure_floor - self.position)
              else:
                  return (self.building.number_of_floors - self.position - 1) + (self.building.number_of_floors - 1) + req.departure_floor
          else:
              return (self.building.number_of_floors - self.position - 1) + (self.building.number_of_floors - req.departure_floor - 1)
      else:
          if req_dir == Elevator.DOWN:
              if req.departure_floor <= self.position:
                  return abs(self.position - req.departure_floor)
              else:
                  return self.position + (self.building.number_of_floors - 1) + (self.building.number_of_floors - req.departure_floor - 1)
          else:
              return self.position + req.departure_floor



    def add_request(self, req):

      direction = (req.arrival_floor - req.departure_floor)/abs(req.arrival_floor - req.departure_floor)

      pickup = PickUp(req.person, req.departure_floor, direction)
      dropoff = DropOff(req.person, req.arrival_floor, direction)
      req.person.wait_time_list.append(Wait(req))

      if self.direction == Elevator.IDLE:
          if pickup.floor == self.position:
              self.wait_time_remaining = Elevator.stopping_time
              self.direction = direction
              req.person.on_elevator = True
          else:
              self.fbas.append(FloorBatchAction([pickup]))
              self.direction = (pickup.floor-self.position)/abs(pickup.floor-self.position)
          self.fbas.append(FloorBatchAction([dropoff]))
      else:
          sequence = [self.direction, -1 * self.direction, self.direction]
          stretch = None
          if pickup.direction != self.direction:
            stretch = 1
          elif self.direction * pickup.floor >= self.direction * self.position:
            stretch = 0
          else:
            stretch = 2
          stretches = [[], [], []]
          i = 0
          while i < len(self.fbas) and self.fbas[i].direction == sequence[0] and sequence[0] * self.fbas[i].floor >= sequence[0] * self.position:
              stretches[0].append(self.fbas[i])
              i += 1
          while i < len(self.fbas) and self.fbas[i].direction == sequence[1]:
              stretches[1].append(self.fbas[i])
              i += 1
          while i < len(self.fbas):
              stretches[2].append(self.fbas[i])
              i += 1

          cur_pos = 0

          if pickup.floor == self.position and self.direction == direction:
              self.wait_time_remaining = Elevator.stopping_time
              req.person.on_elevator = True
              while cur_pos < len(stretches[stretch]) and sequence[stretch] * dropoff.floor > sequence[stretch] * stretches[stretch][cur_pos].floor:
                  cur_pos += 1
              if cur_pos == len(stretches[stretch]) or sequence[stretch] * dropoff.floor < sequence[stretch] * stretches[stretch][cur_pos].floor:
                  stretches[stretch].insert(cur_pos, FloorBatchAction([dropoff]))
              else:
                  stretches[stretch][cur_pos].add_action(dropoff)
              self.fbas = stretches[0] + stretches[1] + stretches[2]
          else:
              while cur_pos < len(stretches[stretch]) and sequence[stretch] * pickup.floor > sequence[stretch] * stretches[stretch][cur_pos].floor:
                  cur_pos += 1
              if cur_pos == len(stretches[stretch]) or sequence[stretch] * pickup.floor < sequence[stretch] * stretches[stretch][cur_pos].floor:
                  stretches[stretch].insert(cur_pos, FloorBatchAction([pickup]))
              else:
                  stretches[stretch][cur_pos].add_action(pickup)
              cur_pos += 1
              while cur_pos < len(stretches[stretch]) and sequence[stretch] * dropoff.floor > sequence[stretch] * stretches[stretch][cur_pos].floor:
                  cur_pos += 1
              if cur_pos == len(stretches[stretch]) or sequence[stretch] * dropoff.floor < sequence[stretch] * stretches[stretch][cur_pos].floor:
                  stretches[stretch].insert(cur_pos, FloorBatchAction([dropoff]))
              else:
                  stretches[stretch][cur_pos].add_action(dropoff)
              self.fbas = stretches[0] + stretches[1] + stretches[2]
      self.next_stop = self.fbas[0]


    # Used in destination dispatch to forecast wait times induced by each elevator.
    def run_to_completion(self):
      while self.direction != Elevator.IDLE:
        self.update_to_next_stop(is_copy=True)

    def wait_time_diff(self, req):
      # Make copy of self to forecast added wait time without changing the state of the system.
      # Temporarily remove building reference to avoid copying unecessary data.
      tmp = self.building
      self.building = None
      without_req = copy.deepcopy(self)
      with_req = copy.deepcopy(self)
      self.building = tmp
      cpy = copy.deepcopy(req)
      with_req.add_request(cpy)

      without_riders = set()
      with_riders = set()

      for fba in without_req.fbas:
          without_riders = without_riders.union(fba.get_people())
      for fba in with_req.fbas:
          with_riders = with_riders.union(fba.get_people())
      without_req.run_to_completion()
      with_req.run_to_completion()
      without_total_wait = sum([per.wait_time_list[-1].total_wait_time for per in without_riders])
      with_total_wait = sum([per.wait_time_list[-1].total_wait_time for per in with_riders])
      return with_total_wait - without_total_wait


class FEL:
    def __init__(self):
        self.events = []

    def enqueue(self, req):
        self.events.append(req)
        self.events = sorted(self.events, key=lambda x: x.time_of_request)

    def dequeue(self):
        ret_val = self.events[0]
        self.events.pop(0)
        return ret_val

    def is_empty(self):
        return len(self.events) == 0

    def peek_front(self):
        return self.events[0]


class Building:

    def __init__(self, number_of_floors, number_of_elevators):
        self.elevators = [Elevator(self) for _ in range(number_of_elevators)]
        self.fel = FEL()
        self.people = []
        self.number_of_floors = number_of_floors
        self.non_lobby_floors = [i for i in range(1, self.number_of_floors)]

        self.clock = 0

    def next_elevator_to_stop(self):
      next_elevator = None
      earliest_time = None
      for elevator in self.elevators:
          if elevator.direction != Elevator.IDLE and (earliest_time == None or elevator.get_next_stop_time() < earliest_time):
              next_elevator = elevator
              earliest_time = elevator.get_next_stop_time()
      return next_elevator


    # Returns sorted list of arrival times based on empirical data.
    def calculate_arrival_times(self, mult):
        x = [i*15 for i in range(13*4+1)]
        # rates from Kuusinen et al.
        rates = [28/60, 56/60, 40/60, 84/60, 148/60, 124/60, 112/60, 120/60, 88/60, 112/60, 80/60, 24/60, 48/60, 12/60, 16/60, 40/60, 40/60, 56/60, 112/60,
                    184/60, 136/60, 100/60, 104/60, 124/60, 56/60, 48/60, 28/60, 20/60, 32/60, 24/60, 56/60, 28/60, 28/60, 0, 24/60, 16/60, 64/60, 4/60, 0,
                    4/60, 8/60, 0, 4/60, 4/60, 0, 4/60, 8/60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        rates = [rate*mult for rate in rates]

        # Smooth measured rates with a rolling average.
        y = []
        for i in range(-2, 13*4-1):
            y.append((2*rates[i%len(rates)]+rates[(i+1)%len(rates)]+2*rates[(i-1)%len(rates)]+rates[(i-2)%len(rates)])/6)

        # Interpolate rates with lines, calculate slope and y-intercept of lines.
        slope_ints = []
        for i in range(13*4):
            m = (y[i+1]-y[i])/(15)
            b = m*(-x[i])+y[i]
            slope_ints.append((m, b))

        # Integrate lines to get quadratics
        quadratics = [(slope_ints[0][0]/2, slope_ints[0][1], 0)]
        for i in range(1, len(slope_ints)):
            quadratics.append((slope_ints[i][0]/2, slope_ints[i][1], quadratics[i-1][0]*(x[i])**2+quadratics[i-1][1]*(x[i])+quadratics[i-1][2]-(slope_ints[i][0]/2*x[i]**2 + slope_ints[i][1]*x[i])))

        inv_endpoints = [0] + [quadratics[i][0]*x[i+1]**2+quadratics[i][1]*x[i+1]+quadratics[i][2] for i in range(len(quadratics))]

        arrivals = []
        next_homogen_arrival = exponential(1)
        cur_end = 1
        # Generate Poisson process with rate 1 and invert the quadratics to generate the NSPP
        while next_homogen_arrival < inv_endpoints[-1]:
            while next_homogen_arrival >= inv_endpoints[cur_end]:
                cur_end += 1
            a = quadratics[cur_end-1][0]
            b = quadratics[cur_end-1][1]
            c = quadratics[cur_end-1][2] - next_homogen_arrival
            if a == 0:
                arrivals.append(-c/b)
            else:
                sol1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
                sol2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)
                if x[cur_end-1] <= sol1 and sol1 < x[cur_end]:
                    arrivals.append(sol1)
                else:
                    arrivals.append(sol2)
            next_homogen_arrival += exponential(1)
        return arrivals


    def initialize_system(self, mult):
        #generate all of the people (init outside), arrival times, departure times, etc
        arrival_times = self.calculate_arrival_times(mult)

        for arrival_time in arrival_times:
            departure_time = arrival_time + normal(AVG_TIME_IN_SYSTEM, SIGMA_TIME_IN_SYSTEM)
            person = Person(arrival_time, departure_time, self.non_lobby_floors)

            #append to people outside building
            self.people.append(person)

            #add initial elevator request to FEL
            self.fel.enqueue(person.generate_initial_request())

    def run_conventional_alg(self, mult=1):

        self.initialize_system(mult)

        while True:
            if not self.fel.is_empty():
                next_request = self.fel.peek_front()

                next_elevator = self.next_elevator_to_stop()
                if next_elevator is not None and next_elevator.get_next_stop_time() < next_request.time_of_request:
                    next_elevator.update_to_next_stop()
                    self.clock = next_elevator.clock
                    continue

                min_dist = None
                chosen_elevator = None

                self.fel.dequeue()
                self.clock = next_request.time_of_request

                for elevator in self.elevators:
                    elevator.update_to_time_t(next_request.time_of_request)
                    distance = elevator.compute_distance(next_request)

                    if min_dist is None or distance < min_dist:
                        min_dist = distance
                        chosen_elevator = elevator

                chosen_elevator.add_request(next_request)


            # If FEL is empty, there may still be people in elevators. Get next_elevator_to_stop()
            # and update it to its next stop, then check again if the FEL is empty.
            else:
                next_elevator = self.next_elevator_to_stop()

                if next_elevator is None:
                  return

                next_elevator.update_to_next_stop()
                self.clock = next_elevator.clock

    def run_destination_dispatch(self, mult=1):

        self.initialize_system(mult)

        while True:
            if not self.fel.is_empty():
                next_request = self.fel.peek_front()

                next_elevator = self.next_elevator_to_stop()
                if next_elevator is not None and next_elevator.get_next_stop_time() < next_request.time_of_request:
                    next_elevator.update_to_next_stop()
                    self.clock = next_elevator.clock
                    continue

                min_diff = None
                chosen_elevator = None

                self.fel.dequeue()
                self.clock = next_request.time_of_request

                for elevator in self.elevators:
                    elevator.update_to_time_t(next_request.time_of_request)
                    cur_diff = elevator.wait_time_diff(next_request)

                    if min_diff is None or cur_diff < min_diff:
                        min_diff = cur_diff
                        chosen_elevator = elevator

                chosen_elevator.add_request(next_request)


            # If FEL is empty, there may still be people in elevators. Get next_elevator_to_stop()
            # and update it to its next stop, then check again if the FEL is empty.
            else:
                next_elevator = self.next_elevator_to_stop()

                if next_elevator is None:
                  return

                next_elevator.update_to_next_stop()
                self.clock = next_elevator.clock