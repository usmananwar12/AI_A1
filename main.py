from typing import List, Tuple
import heapq
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class SearchAlgorithm:

    @staticmethod
    def get_neighbors(x: int, y: int, grid: List[List[str]]) -> List[Tuple[int, int]]:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != '-1':
                neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def get_start_target(grid: List[List[str]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start = target = None
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 's':
                    start = (r, c)
                elif grid[r][c] == 't':
                    target = (r, c)
        return start, target

    @staticmethod
    def best_first_search(grid: List[List[str]]) -> Tuple[int, List[Tuple[int, int]]]:
        start, target = SearchAlgorithm.get_start_target(grid)
        pq = [(SearchAlgorithm.heuristic(start, target), start)]
        came_from = {start: None}

        while pq:
            _, current = heapq.heappop(pq)
            if current == target:
                return 1, SearchAlgorithm.reconstruct_path(came_from, start, target)
            for neighbor in SearchAlgorithm.get_neighbors(*current, grid):
                if neighbor not in came_from:
                    heapq.heappush(pq, (SearchAlgorithm.heuristic(neighbor, target), neighbor))
                    came_from[neighbor] = current
        return -1, []

    @staticmethod
    def heuristic(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def reconstruct_path(came_from, start, target):
        path = []
        current = target
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]

    @staticmethod
    def plot_grid(grid, path):
        grid_array = np.array([[1 if cell == '-1' else 0 for cell in row] for row in grid])
        for r, c in path:
            grid_array[r][c] = 2  # Mark path

        plt.figure(figsize=(6, 6))
        sns.heatmap(grid_array, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black', cbar=False)
        plt.title("Best First Search Path Visualization")
        plt.show()


if __name__ == "__main__":
    example = [
        ['0', '0', '0', '0'],
        ['0', '-1', '-1', 't'],
        ['s', '0', '-1', '0'],
        ['0', '0', '0', '-1']
    ]

    found, path = SearchAlgorithm.best_first_search(example)
    print("Found:", found)
    print("Path:", path)

    if found == 1:
        SearchAlgorithm.plot_grid(example, path)
