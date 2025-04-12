#!/usr/bin/env python3
"""
Main entry point for the archaeological analysis application.
This module provides a command-line interface for accessing various analysis tools.
"""

import inquirer
import sys
from vasijas import vasijas
from glifos import glifos
from glifos import variantes
from terrenos import terreno

def display_menu():
    """
    Display the main menu and return the user's selection.
    
    Returns:
        str: The selected option
    """
    questions = [
        inquirer.List(
            'opcion',
            message="¿Qué desea realizar?",
            choices=[
                '1. Análisis de glifos',
                '2. Reconocimiento de vasijas',
                '3. Reconocimiento de terrenos',
                '4. Salir'
            ],
        ),
    ]
    
    return inquirer.prompt(questions)['opcion']

def handle_glifos():
    """Handle the glyph analysis option."""
    print("Ejecutando análisis de glifos")
    
    # Display submenu for glyph analysis
    questions = [
        inquirer.List(
            'glifos_option',
            message="Seleccione una opción:",
            choices=[
                '1. Entrenar modelo',
                '2. Predecir glifos',
                '3. Generar variantes',
                '4. Volver al menú principal'
            ],
        ),
    ]
    
    glifos_option = inquirer.prompt(questions)['glifos_option']
    
    if glifos_option == '1. Entrenar modelo':
        glifos.train()
    elif glifos_option == '2. Predecir glifos':
        glifos.predict()
    elif glifos_option == '3. Generar variantes':
        # Ask for input and output directories and number of variants
        questions = [
            inquirer.Text('input_dir', message="Directorio de entrada:"),
            inquirer.Text('output_dir', message="Directorio de salida:"),
            inquirer.Text('num_variants', message="Número de variantes a generar:")
        ]
        
        answers = inquirer.prompt(questions)
        try:
            num_variants = int(answers['num_variants'])
            variantes.process_directory(
                answers['input_dir'], 
                answers['output_dir'], 
                num_variants
            )
        except ValueError:
            print("Error: El número de variantes debe ser un entero.")
    
    # Return to main menu after completing the operation
    main()

def handle_vasijas():
    """Handle the vessel recognition option."""
    print("Ejecutando reconocimiento de vasijas")
    vasijas.main()
    main()

def handle_terrenos():
    """Handle the terrain recognition option."""
    print("Ejecutando reconocimiento de terrenos")
    
    # Create a Terreno instance
    terreno_obj = terreno.Terreno(name="Análisis de terreno")
    
    # Display submenu for terrain analysis
    questions = [
        inquirer.List(
            'terreno_option',
            message="Seleccione una opción:",
            choices=[
                '1. Cargar imagen',
                '2. Detectar estructuras',
                '3. Visualizar resultados',
                '4. Exportar datos',
                '5. Volver al menú principal'
            ],
        ),
    ]
    
    terreno_option = inquirer.prompt(questions)['terreno_option']
    
    if terreno_option == '1. Cargar imagen':
        # Ask for image path
        questions = [
            inquirer.Text('image_path', message="Ruta de la imagen satelital:")
        ]
        
        image_path = inquirer.prompt(questions)['image_path']
        if terreno_obj.load_image(image_path):
            print(f"Imagen cargada correctamente: {image_path}")
            # Ask if user wants to preprocess the image
            if inquirer.confirm("¿Desea preprocesar la imagen?", default=True):
                terreno_obj.preprocess()
                print("Imagen preprocesada correctamente")
        else:
            print(f"Error al cargar la imagen: {image_path}")
    
    elif terreno_option == '2. Detectar estructuras':
        if terreno_obj.image is None:
            print("Error: Primero debe cargar una imagen")
        else:
            # Ask for detection method
            questions = [
                inquirer.List(
                    'detection_method',
                    message="Seleccione el método de detección:",
                    choices=[
                        '1. Detección por umbral',
                        '2. Detección por color',
                        '3. Segmentación'
                    ],
                ),
            ]
            
            detection_method = inquirer.prompt(questions)['detection_method']
            
            if detection_method == '1. Detección por umbral':
                # Ask for threshold method
                questions = [
                    inquirer.List(
                        'threshold_method',
                        message="Seleccione el método de umbral:",
                        choices=['adaptive', 'otsu', 'binary'],
                    ),
                ]
                
                threshold_method = inquirer.prompt(questions)['threshold_method']
                terreno_obj.detect_structures(threshold_method=threshold_method)
                print(f"Se detectaron {len(terreno_obj.structures)} estructuras")
            
            elif detection_method == '2. Detección por color':
                # Ask for target color
                questions = [
                    inquirer.Text('target_color', message="Color objetivo (formato HEX, ej: #FF0000):")
                ]
                
                target_color = inquirer.prompt(questions)['target_color']
                terreno_obj.detect_structures_by_color(target_color)
                print(f"Se detectaron {len(terreno_obj.structures)} estructuras por color")
            
            elif detection_method == '3. Segmentación':
                # Ask for number of segments
                questions = [
                    inquirer.Text('num_segments', message="Número de segmentos:")
                ]
                
                try:
                    num_segments = int(inquirer.prompt(questions)['num_segments'])
                    segmented = terreno_obj.segment_image(num_segments=num_segments)
                    if segmented is not None:
                        print(f"Imagen segmentada en {num_segments} segmentos")
                except ValueError:
                    print("Error: El número de segmentos debe ser un entero.")
    
    elif terreno_option == '3. Visualizar resultados':
        if not terreno_obj.structures:
            print("Error: No hay estructuras detectadas para visualizar")
        else:
            # Ask for output path
            questions = [
                inquirer.Text('output_path', message="Ruta para guardar la visualización (opcional):")
            ]
            
            output_path = inquirer.prompt(questions)['output_path']
            output_path = output_path if output_path.strip() else None
            
            terreno_obj.visualize_structures(output_path=output_path)
            
            # Display statistics
            stats = terreno_obj.get_structure_statistics()
            print("\nEstadísticas de estructuras:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    elif terreno_option == '4. Exportar datos':
        if not terreno_obj.structures:
            print("Error: No hay estructuras detectadas para exportar")
        else:
            # Ask for output path
            questions = [
                inquirer.Text('output_path', message="Ruta para guardar los datos:")
            ]
            
            output_path = inquirer.prompt(questions)['output_path']
            if terreno_obj.export_structure_data(output_path):
                print(f"Datos exportados correctamente a: {output_path}")
            else:
                print(f"Error al exportar los datos")
    
    # Return to main menu after completing the operation
    main()

def main():
    """Main function that displays the menu and handles user selection."""
    try:
        option = display_menu()
        
        if option == '1. Análisis de glifos':
            handle_glifos()
        elif option == '2. Reconocimiento de vasijas':
            handle_vasijas()
        elif option == '3. Reconocimiento de terrenos':
            handle_terrenos()
        elif option == '4. Salir':
            print("Saliendo del programa")
            sys.exit(0)
        else:
            print("Opción incorrecta")
            main()
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
