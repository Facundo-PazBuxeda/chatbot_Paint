from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import mysql.connector
import time

# Configuración de Selenium
driver = webdriver.Chrome()
base_url = "https://somosrex.com/pinturas-y-accesorios.html"
max_pages = 50

# Conexión a MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="pintureria"
)

mycursor = db.cursor()

# Crear la tabla si no existe
mycursor.execute('''
CREATE TABLE IF NOT EXISTS productos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    marca VARCHAR(255),
    nombre VARCHAR(255),
    precio_regular VARCHAR(255),
    precio_promo VARCHAR(255)
)
''')
db.commit()

productos_encontrados = 0

# Abre la página inicial
driver.get(base_url)

# Espera para resolver el CAPTCHA manualmente
print("Por favor, resuelve el CAPTCHA en el navegador.")
time.sleep(60)  # Asegúrate de que el CAPTCHA esté resuelto
# Itera sobre las páginas
for page in range(1, max_pages + 1):
    print(f"Scrapeando la página {page}...")
    url = f"{base_url}?p={page}"
    driver.get(url)
    time.sleep(5)

    # Extraer productos usando Selenium
    items = driver.find_elements(By.CSS_SELECTOR, 'li.item.product.product-item')
    print(f"Productos encontrados en página {page}: {len(items)}")

    for item in items:
        try:
            # Marca
            try:
                marca = driver.execute_script("""
                            return arguments[0].querySelector('span.brand-name').textContent;
                        """, item).strip()
            except NoSuchElementException:
                marca = 'No especificada'

            # Nombre
            try:
                nombre = item.find_element(By.CSS_SELECTOR, 'a.product-item-link').text.strip()
            except NoSuchElementException:
                nombre = 'No especificado'

            # Precio regular
            try:
                precio_regular = item.find_element(By.CSS_SELECTOR, 'span[data-price-type="oldPrice"]').text.strip()
            except NoSuchElementException:
                precio_regular = None

            # Precio promocional
            try:
                precio_promo = item.find_element(By.CSS_SELECTOR, 'span[data-price-type="defaultPromoPrice"]').text.strip()
            except NoSuchElementException:
                try:
                    precio_promo = item.find_element(By.CSS_SELECTOR, 'span[data-price-type="finalPrice"]').text.strip()
                except NoSuchElementException:
                    precio_promo = 'No especificado'

            # Insertar en la base de datos
            if marca != 'No especificada' or nombre != 'No especificado':
                sql = '''INSERT INTO productos (marca, nombre, precio_regular, precio_promo) 
                        VALUES (%s, %s, %s, %s)'''
                valores = (marca, nombre, precio_regular, precio_promo)
                mycursor.execute(sql, valores)
                db.commit()
                productos_encontrados += 1
                print(f"Producto insertado: {nombre}")

        except Exception as e:
            print(f"Error procesando producto: {e}")
            continue

db.close()
driver.quit()

print(f"Proceso completado. Total de productos almacenados: {productos_encontrados}")
