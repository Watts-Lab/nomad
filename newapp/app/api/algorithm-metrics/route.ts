import { NextRequest, NextResponse } from "next/server"
import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

export const runtime = "nodejs"

type MetricsPayload = {
  action?: "options" | "run"
  mode?: "single" | "compare"
  left_algorithm?: string
  right_algorithm?: string
  trajectory_user_id?: string
  iterations?: number
  sweep_min?: number
  sweep_max?: number
  sweep_points?: number
  seed?: number
}

function runPythonMetrics(payload: MetricsPayload): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "scripts", "algorithm_metrics.py")
    const preferredPython = "/opt/anaconda3/bin/python"
    const pythonBin =
      process.env.PYTHON_BIN ||
      process.env.PYTHON ||
      (fs.existsSync(preferredPython) ? preferredPython : "python3")

    const proc = spawn(pythonBin, [scriptPath], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        MPLCONFIGDIR: process.env.MPLCONFIGDIR || "/tmp/matplotlib-cache",
      },
    })

    let stdout = ""
    let stderr = ""

    proc.stdout.on("data", (chunk) => {
      stdout += chunk.toString()
    })

    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString()
    })

    proc.on("error", (err) => {
      reject(new Error(`Failed to start python process: ${err.message}`))
    })

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python exited with code ${code}. ${stderr}`))
        return
      }
      try {
        resolve(JSON.parse(stdout))
      } catch (err) {
        reject(new Error(`Invalid JSON from python. ${String(err)}. stderr: ${stderr}`))
      }
    })

    proc.stdin.write(JSON.stringify(payload))
    proc.stdin.end()
  })
}

export async function GET() {
  try {
    const data = await runPythonMetrics({ action: "options" })
    return NextResponse.json(data)
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Unknown metrics options error" },
      { status: 500 },
    )
  }
}

export async function POST(req: NextRequest) {
  try {
    const payload = (await req.json()) as MetricsPayload
    const data = await runPythonMetrics({ ...payload, action: "run" })
    return NextResponse.json(data)
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Unknown metrics run error" },
      { status: 500 },
    )
  }
}
