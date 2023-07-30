from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
from PIL import Image
import numpy as np
from tqdm import tqdm


def image_to_qubits(image_path, qubits_per_block=2):
    # Load the image and convert it to grayscale
    img = Image.open(image_path).convert("L")

    # Resize the image to a square of size 512x512
    img = img.resize((512, 512))

    # Get the pixel values as a numpy array
    pixel_data = np.array(img)

    # Divide the image into smaller blocks of size qubits_per_block x qubits_per_block
    num_blocks = 512 // qubits_per_block
    blocks = [
        pixel_data[i : i + qubits_per_block, j : j + qubits_per_block]
        for i in range(0, 512, qubits_per_block)
        for j in range(0, 512, qubits_per_block)
    ]

    # Normalize the pixel values to be between 0 and 1
    blocks = [block / 255 for block in blocks]

    # Initialize a list to store qubits for each block
    qubit_list = []

    # Create a quantum circuit for each block and encode the pixel values as qubits
    for block in tqdm(blocks, desc="Converting to qubits"):
        num_qubits = qubits_per_block ** 2
        qc = QuantumCircuit(num_qubits)

        for i in range(qubits_per_block):
            for j in range(qubits_per_block):
                # Encode pixel values as qubits using amplitude encoding
                theta = 2 * np.arcsin(np.sqrt(block[i, j]))
                qc.ry(theta, i * qubits_per_block + j)

        qubit_list.append(qc)
    print("Image converted to qubits")
    return qubit_list


def qubits_to_image(qubit_list, qubits_per_block=2):
    # Initialize an empty array to store the pixel values
    pixel_data = np.zeros((512, 512))

    # Retrieve the pixel values from the qubits and decode them
    for i, qc in tqdm(enumerate(qubit_list), desc="Retrieving image"):
        job = execute(qc, Aer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        
        for j in range(qubits_per_block):
            for k in range(qubits_per_block):
                # Convert the qubit amplitudes back to pixel values
                pixel_value = np.sin(np.arcsin(np.real(statevector[j * qubits_per_block + k])) ** 2) * 255
                # Update the pixel_data array using the correct indices
                pixel_data[(i // (512 // qubits_per_block)) + j, (i % (512 // qubits_per_block)) + k] = pixel_value

    # Create and save the image from the pixel data
    new_size = 512//qubits_per_block
    pixel_data = pixel_data[:new_size, :new_size]
    retrieved_image = Image.fromarray(pixel_data.astype('uint8'))
    retrieved_image.show()



# Test the functions with a sample image
image_path = input("Enter the path to the image: ")
qubits_per_block = int(input("Enter the number of qubits per block: ")) # Recommended value: 2

# Transform image into qubits
qubit_list = image_to_qubits(image_path, qubits_per_block)

# Retrieve the image from the qubits
qubits_to_image(qubit_list, qubits_per_block)

