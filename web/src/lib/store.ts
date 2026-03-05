"use client";

import { create } from "zustand";

// TRM State Types
export interface RecursionState {
    x: number[][]; // Input puzzle
    y: number[][]; // Current solution
    z: number[][][]; // Latent traces per cycle
    cycle: number;
    isRunning: boolean;
    accuracy: number;
    haltProbability: number;
}

export interface TRMStore {
    // Sudoku state
    sudoku: {
        puzzle: number[];
        solution: number[];
        prediction: number[];
        isLoading: boolean;
    };

    // Recursion visualization
    recursion: RecursionState;

    // Z-trace nodes for React Flow
    zNodes: ZNode[];
    zEdges: ZEdge[];

    // Actions
    setPuzzle: (puzzle: number[], solution: number[]) => void;
    setPrediction: (prediction: number[]) => void;
    setLoading: (loading: boolean) => void;
    updateRecursion: (cycle: number, y: number[][], z: number[][][]) => void;
    addZNode: (node: ZNode) => void;
    resetRecursion: () => void;
}

export interface ZNode {
    id: string;
    type: "input" | "latent" | "output" | "cycle";
    position: { x: number; y: number };
    data: {
        label: string;
        value?: number;
        confidence?: number;
        cycle?: number;
    };
}

export interface ZEdge {
    id: string;
    source: string;
    target: string;
    animated?: boolean;
    style?: Record<string, string>;
}

// Initial state
const initialRecursion: RecursionState = {
    x: [],
    y: [],
    z: [],
    cycle: 0,
    isRunning: false,
    accuracy: 0,
    haltProbability: 0,
};

// Zustand store
export const useTRMStore = create<TRMStore>((set) => ({
    sudoku: {
        puzzle: [],
        solution: [],
        prediction: [],
        isLoading: false,
    },

    recursion: initialRecursion,
    zNodes: [],
    zEdges: [],

    setPuzzle: (puzzle, solution) =>
        set((state) => ({
            sudoku: { ...state.sudoku, puzzle, solution, prediction: [] },
        })),

    setPrediction: (prediction) =>
        set((state) => ({
            sudoku: { ...state.sudoku, prediction },
        })),

    setLoading: (loading) =>
        set((state) => ({
            sudoku: { ...state.sudoku, isLoading: loading },
        })),

    updateRecursion: (cycle, y, z) =>
        set((state) => ({
            recursion: { ...state.recursion, cycle, y, z, isRunning: true },
        })),

    addZNode: (node) =>
        set((state) => ({
            zNodes: [...state.zNodes, node],
        })),

    resetRecursion: () =>
        set({
            recursion: initialRecursion,
            zNodes: [],
            zEdges: [],
        }),
}));

// Hook for TRM recursion simulation
export function useRecursion(
    x: number[],
    y_init: number[],
    T_cycles: number = 3
) {
    const { updateRecursion, setPrediction, setLoading } = useTRMStore();

    const runRecursion = async () => {
        setLoading(true);

        let y = [...y_init];
        const zTraces: number[][][] = [];

        for (let t = 0; t < T_cycles; t++) {
            // Simulate latent updates
            const z: number[][] = [];
            for (let i = 0; i < 6; i++) {
                // n_latent = 6
                const zStep = y.map((v, idx) =>
                    x[idx] !== 0 ? x[idx] : Math.floor(Math.random() * 9) + 1
                );
                z.push(zStep);
                await new Promise((r) => setTimeout(r, 50));
            }

            // y-refine
            y = y.map((v, idx) => (x[idx] !== 0 ? x[idx] : z[z.length - 1][idx]));
            zTraces.push(z);

            updateRecursion(t + 1, [y], zTraces);
            await new Promise((r) => setTimeout(r, 100));
        }

        setPrediction(y);
        setLoading(false);

        return y;
    };

    return { runRecursion };
}
