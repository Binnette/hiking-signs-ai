import torch

devices = [
    "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", 
    "ve", "fpga", "maia", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia", "privateuseone"
]

available_devices = []

def check_device(device):
    if device == "cpu":
        return True
    elif device == "cuda":
        return torch.cuda.is_available()
    elif device == "ipu":
        return hasattr(torch, 'ipu')  # Placeholder as PyTorch IPU support may not be available
    elif device == "xpu":
        return hasattr(torch, 'xpu')  # Placeholder for future support
    elif device == "mkldnn":
        return hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available()
    elif device == "opengl":
        return hasattr(torch, 'opengl')  # Placeholder for future support
    elif device == "opencl":
        return hasattr(torch, 'opencl')  # Placeholder for future support
    elif device == "ideep":
        return hasattr(torch.backends, 'ideep') and torch.backends.ideep.is_available()
    elif device == "hip":
        return hasattr(torch, 'hip') and torch.hip.is_available()
    elif device == "ve":
        return hasattr(torch, 've')  # Placeholder for future support
    elif device == "fpga":
        return hasattr(torch, 'fpga')  # Placeholder for future support
    elif device == "maia":
        return hasattr(torch, 'maia')  # Placeholder for future support
    elif device == "xla":
        return hasattr(torch, 'xla')  # Placeholder for future support
    elif device == "lazy":
        return hasattr(torch, 'lazy')  # Placeholder for future support
    elif device == "vulkan":
        return hasattr(torch, 'vulkan')  # Placeholder for future support
    elif device == "mps":
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    elif device == "meta":
        return hasattr(torch, 'meta')  # Placeholder for future support
    elif device == "hpu":
        return hasattr(torch, 'hpu')  # Placeholder for future support
    elif device == "mtia":
        return hasattr(torch, 'mtia')  # Placeholder for future support
    elif device == "privateuseone":
        return hasattr(torch, 'privateuseone')  # Placeholder for future support
    return False

for device in devices:
    if check_device(device):
        available_devices.append(device)
        print(f"Device {device} is available.")
    else:
        print(f"Device {device} is not available.")

if not available_devices:
    print("No specified devices are available.")
else:
    print("Available devices:", available_devices)
