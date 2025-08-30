// Three.js background with particle field and subtle motion
import * as THREE from 'https://unpkg.com/three@0.159.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.159.0/examples/jsm/controls/OrbitControls.js';

const canvas = document.getElementById('bg');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.6, 3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 1.5;
controls.maxDistance = 6;

// Lights
const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(2, 3, 4);
scene.add(light);
scene.add(new THREE.AmbientLight(0x8899aa, 0.6));

// Particles
const count = 3000;
const geo = new THREE.BufferGeometry();
const positions = new Float32Array(count * 3);
for (let i = 0; i < count * 3; i+=3){
  const r = 6 * Math.random() + 1;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos((Math.random()*2)-1);
  positions[i] = r * Math.sin(phi) * Math.cos(theta);
  positions[i+1] = r * Math.sin(phi) * Math.sin(theta);
  positions[i+2] = r * Math.cos(phi);
}
geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const mat = new THREE.PointsMaterial({ size: 0.015, sizeAttenuation: true, color: 0x96ff85, transparent: true, opacity: 0.75 });
const points = new THREE.Points(geo, mat);
scene.add(points);

// Animate
let t = 0;
function animate(){
  requestAnimationFrame(animate);
  t += 0.0015;
  points.rotation.y = t;
  controls.update();
  renderer.render(scene, camera);
}
animate();

// Resize
window.addEventListener('resize', ()=>{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Respect reduced motion: stop rotation if user prefers less motion
const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)');
function handleReduce(){
  if (prefersReduced.matches){
    points.rotation.y = 0;
    animate = function(){ renderer.render(scene, camera); };
  }
}
prefersReduced.addEventListener?.('change', handleReduce);
handleReduce();
