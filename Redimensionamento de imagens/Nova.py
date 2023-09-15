from PIL import Image

def resize_image_bicubic(image, new_width, new_height):
    # Redimensionar a imagem usando interpolação bicúbica
    resized_image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    return resized_image

# Carregar a imagem original
image = Image.open('teste.jpg')

# Redimensionar a imagem usando interpolação bicúbica
resized_image = resize_image_bicubic(image, 400, 400)

# Salvar a imagem redimensionada
resized_image.save('testado.jpg')

# Exibir a mensagem de confirmação
print("Imagem redimensionada foi salva com sucesso.")
