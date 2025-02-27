from PIL import Image, ImageDraw

def generate_grid_image(grid_size, tile_size, output_file):
    width = grid_size * tile_size
    height = grid_size * tile_size

    # Create a blank white image
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Draw grid lines
    for x in range(0, width, tile_size):
        draw.line([(x, 0), (x, height)], fill="black")
    for y in range(0, height, tile_size):
        draw.line([(0, y), (width, y)], fill="black")

    image.save(output_file)
    print(f"Grid image saved to {output_file}")

if __name__ == "__main__":
    generate_grid_image(grid_size=15, tile_size=32, output_file="grid.png")
